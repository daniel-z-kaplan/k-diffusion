import torch
from torch import FloatTensor, BoolTensor, LongTensor, distributed as dist
from accelerate import Accelerator
from typing import Literal

from .precompute_conds import PrecomputedConds
from .crossattn_extra_args import CrossAttnExtraArgs
from .crossattn_cfg_args import CrossAttnCFGArgs

def get_precomputed_cond_cfg_args(
    accelerator: Accelerator,
    precomputed_conds: PrecomputedConds,
    demo_uncond: Literal['allzeros', 'emptystr'],
    n_per_proc: int,
    demo_gen: torch.Generator,
) -> CrossAttnCFGArgs:
    caption_ix: LongTensor = torch.randint(0, precomputed_conds.masked_conds.cond.shape[0], [accelerator.num_processes, n_per_proc], generator=demo_gen).to(accelerator.device)
    dist.broadcast(caption_ix, 0)
    cond: FloatTensor = precomputed_conds.masked_conds.cond.index_select(0, caption_ix[accelerator.process_index])
    cond_mask: BoolTensor = precomputed_conds.masked_conds.mask.index_select(0, caption_ix[accelerator.process_index])
    if demo_uncond == 'allzeros':
        cond[caption_ix[accelerator.process_index] == precomputed_conds.text_uncond_ix] = 0
        cond_mask[caption_ix[accelerator.process_index] == precomputed_conds.text_uncond_ix] = 1
        masked_uncond = precomputed_conds.allzeros_masked_uncond
    elif demo_uncond == 'emptystr':
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