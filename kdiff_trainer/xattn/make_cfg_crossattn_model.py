import torch
from torch import FloatTensor, BoolTensor

def make_cfg_crossattn_model_fn(model, xuncond: FloatTensor, xuncond_mask: BoolTensor, cfg_scale: int):
    def cfg_model_fn(x, sigma, xcond: FloatTensor, xcond_mask: BoolTensor):
        x_in = torch.cat([x, x])
        sigma_in = torch.cat([sigma, sigma])
        xcond_in = torch.cat([xuncond, xcond])
        xcond_mask_in = torch.cat([xuncond_mask, xcond_mask])
        out: FloatTensor = model(x_in, sigma_in, crossattn_cond=xcond_in, crossattn_mask=xcond_mask_in)
        out_uncond, out_cond = out.chunk(2)
        return out_uncond + (out_cond - out_uncond) * cfg_scale
    if cfg_scale != 1:
        return cfg_model_fn
    return model