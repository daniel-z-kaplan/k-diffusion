import torch
from torch import FloatTensor, BoolTensor, LongTensor, distributed as dist
from accelerate import Accelerator
from typing import Literal, Optional, List

from .precompute_conds import PrecomputedConds
from .crossattn_extra_args import CrossAttnExtraArgs
from .crossattn_cfg_args import CrossAttnCFGArgs

def get_precomputed_cond_cfg_args(
    accelerator: Accelerator,
    precomputed_conds: PrecomputedConds,
    uncond_class_ix: int,
    uncond_type: Literal['allzeros', 'emptystr'],
    n_per_proc: int,
    distribute: bool,
    rng: Optional[torch.Generator] = None,
) -> CrossAttnCFGArgs:
    captions_shape: List[int] = [accelerator.num_processes, n_per_proc] if distribute else [n_per_proc]
    caption_ix: LongTensor = torch.randint(0, precomputed_conds.masked_conds.cond.shape[0], captions_shape, generator=rng).to(accelerator.device)
    if distribute:
        dist.broadcast(caption_ix, 0)
        my_captions: LongTensor = caption_ix[accelerator.process_index]
    else:
        my_captions: LongTensor = caption_ix
    cond: FloatTensor = precomputed_conds.masked_conds.cond.index_select(0, my_captions)
    cond_mask: BoolTensor = precomputed_conds.masked_conds.mask.index_select(0, my_captions)
    if uncond_type == 'allzeros':
        cond[my_captions == uncond_class_ix] = 0
        cond_mask[my_captions == uncond_class_ix] = 1
        masked_uncond = precomputed_conds.allzeros_masked_uncond
    elif uncond_type == 'emptystr':
        masked_uncond = precomputed_conds.emptystr_masked_uncond
    sampling_extra_args = CrossAttnExtraArgs(
        crossattn_cond=cond,
        crossattn_mask=cond_mask,
    )
    return CrossAttnCFGArgs(
        masked_uncond=masked_uncond,
        sampling_extra_args=sampling_extra_args,
        caption_ix=caption_ix,
    )