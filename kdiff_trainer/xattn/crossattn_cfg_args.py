from torch import LongTensor
from typing import NamedTuple

from .masked_cond import MaskedCond
from .crossattn_extra_args import CrossAttnExtraArgs

class CrossAttnCFGArgs(NamedTuple):
  masked_uncond: MaskedCond
  sampling_extra_args: CrossAttnExtraArgs
  caption_ix: LongTensor