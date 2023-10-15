import torch
from torch import FloatTensor, BoolTensor, LongTensor

from .precompute_conds import PrecomputedConds
from .crossattn_extra_args import CrossAttnExtraArgs

def get_precomputed_conds_by_ix(
    device: torch.device,
    precomputed_conds: PrecomputedConds,
    embed_ix: LongTensor,
    cond_dropout_rate: float,
    allzeros_uncond_rate: float,
) -> CrossAttnExtraArgs:
    drop = torch.rand(embed_ix.shape[0], device=device)
    batch_text_embeds: FloatTensor = precomputed_conds.masked_conds.cond.index_select(0, embed_ix)
    batch_text_embeds[drop < cond_dropout_rate] = precomputed_conds.masked_conds.cond[precomputed_conds.text_uncond_ix]
    batch_text_embeds[drop < cond_dropout_rate * allzeros_uncond_rate] = 0

    batch_token_masks: BoolTensor = precomputed_conds.masked_conds.mask.index_select(0, embed_ix)
    batch_token_masks[drop < cond_dropout_rate] = precomputed_conds.masked_conds.mask[precomputed_conds.text_uncond_ix]
    batch_token_masks[drop < cond_dropout_rate * allzeros_uncond_rate] = 1

    return CrossAttnExtraArgs(
        crossattn_cond=batch_text_embeds,
        crossattn_mask=batch_token_masks,
    )