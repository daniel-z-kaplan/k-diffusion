from typing import List
import torch
from torch import LongTensor, FloatTensor, BoolTensor, inference_mode
from torch import distributed as dist
from torch.distributed import Work
from accelerate import Accelerator
from argparse import Namespace
from dataclasses import dataclass
import gc

@dataclass
class PrecomputedConds:
    text_uncond_ix: int
    class_captions: List[str]
    text_embeds: FloatTensor
    token_mask: BoolTensor
    emptystr_uncond: FloatTensor
    emptystr_uncond_mask: BoolTensor
    allzeros_uncond: FloatTensor
    allzeros_uncond_mask: BoolTensor

def precompute_conds(
    accelerator: Accelerator,
    dataset_config,
    args: Namespace,
) -> PrecomputedConds:
    assert dataset_config['classes_to_captions'] == 'oxford-flowers'
    from transformers import CLIPTextConfig
    from kdiff_trainer.dataset_meta.oxford_flowers import flower_classes
    text_uncond_ix = 0
    uncond = ''
    class_captions: List[str] = [uncond, *flower_classes]

    text_model_name = 'openai/clip-vit-large-patch14'
    text_config: CLIPTextConfig = CLIPTextConfig.from_pretrained(
        text_model_name,
        cache_dir=args.text_model_hf_cache_dir,
    )
    max_length: int = text_config.max_position_embeddings

    expected_embed_shape = torch.Size((len(class_captions), max_length, text_config.hidden_size))
    expected_mask_shape = torch.Size((len(class_captions), max_length))

    match(accelerator.mixed_precision):
        case 'bf16':
            embed_dtype = torch.bfloat16
        case 'fp16':
            embed_dtype = torch.float16
        case 'fp8':
            # seriously?
            embed_dtype = torch.float8_e4m3fn
        case _:
            embed_dtype = torch.float32

    if accelerator.is_main_process:
        from transformers import CLIPTextModel, CLIPTokenizerFast
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, TensorType
        text_model: CLIPTextModel = CLIPTextModel.from_pretrained(
            text_model_name,
            config=text_config,
            cache_dir=args.text_model_hf_cache_dir,
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to(accelerator.device).eval()
        tokenizer: CLIPTokenizerFast = CLIPTokenizerFast.from_pretrained(text_model_name)
        tokens_out: BatchEncoding = tokenizer(
            class_captions,
            return_tensors=TensorType.PYTORCH,
            padding=PaddingStrategy.MAX_LENGTH,
            max_length=max_length,
            return_attention_mask=True,
            return_length=True,
            add_special_tokens=True,
        )
        tokens: LongTensor = tokens_out['input_ids'].to(accelerator.device)
        token_mask: LongTensor = tokens_out['attention_mask'].to(accelerator.device, dtype=torch.bool)
        del tokens_out, tokenizer
        assert token_mask.shape == expected_mask_shape
        # https://github.com/openai/CLIP/issues/183
        # supposedly we shouldn't supply the token mask to CLIPTextEncoder -- OpenAI say
        # that the causal mask is already enough to protect you from attending to padding?
        # in fact, I certainly noticed with SDXL that supplying mask to CLIPTextEncoder gave me bad results:
        # https://github.com/Birch-san/sdxl-play/blob/afabe5d173553511d0fd0d65c34dffb234745e69/src/embed_mgmt/embed.py#L24
        with inference_mode():
            encoder_out: BaseModelOutputWithPooling = text_model.forward(
                tokens,
                # attention_mask=token_mask,
                # we need it to give us access to penultimate hidden states
                output_hidden_states=True,
                return_dict=True,
            )
            del tokens
            # these are penultimate hidden states
            text_embeds: FloatTensor = encoder_out.hidden_states[-2].to(embed_dtype)
            assert text_embeds.shape == expected_embed_shape
            del encoder_out, text_model
    else:
        text_embeds: FloatTensor = torch.empty(expected_embed_shape, dtype=embed_dtype, device=accelerator.device)
        token_mask: BoolTensor = torch.empty(expected_mask_shape, dtype=torch.bool, device=accelerator.device)
    emb_handle: Work = dist.broadcast(text_embeds, 0, async_op=True)
    mask_handle: Work = dist.broadcast(token_mask, 0, async_op=True)
    emb_handle.wait()
    mask_handle.wait()
    emptystr_uncond: FloatTensor = text_embeds[text_uncond_ix].unsqueeze(0)
    emptystr_uncond_mask: BoolTensor = token_mask[text_uncond_ix].unsqueeze(0)
    allzeros_uncond: FloatTensor = torch.zeros_like(emptystr_uncond)
    allzeros_uncond_mask: BoolTensor = torch.ones_like(emptystr_uncond_mask)

    # loading a language model into VRAM may cause awkward fragmentation, so let's free any buffers it reserved
    gc.collect()
    torch.cuda.empty_cache()

    return PrecomputedConds(
        text_uncond_ix=text_uncond_ix,
        class_captions=class_captions,
        text_embeds=text_embeds,
        token_mask=token_mask,
        emptystr_uncond=emptystr_uncond,
        emptystr_uncond_mask=emptystr_uncond_mask,
        allzeros_uncond=allzeros_uncond,
        allzeros_uncond_mask=allzeros_uncond_mask,
    )