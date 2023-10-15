#!/usr/bin/env python3

"""Trains Karras et al. (2022) diffusion models."""

import argparse
from copy import deepcopy
from functools import partial
import importlib.util
import math
import json
from pathlib import Path
import time
from os import makedirs
from os.path import relpath

import accelerate
import safetensors.torch as safetorch
import torch
import torch._dynamo
from torch import distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch import multiprocessing as mp
from torch import optim, FloatTensor, LongTensor
from torch.utils import data
from torchvision import datasets, transforms, utils
from tqdm.auto import tqdm
from typing import List, Optional
from PIL import Image, ImageFont

import k_diffusion as K
from k_diffusion.utils import DataSetTransform, BatchData
from kdiff_trainer.make_captioned_grid import GridCaptioner, BBox, Typesetting, make_grid_captioner, make_typesetting
from kdiff_trainer.xattn.precompute_conds import precompute_conds, PrecomputedConds
from kdiff_trainer.xattn.precomputed_cond_cfg_args import get_precomputed_cond_cfg_args
from kdiff_trainer.xattn.make_cfg_crossattn_model import make_cfg_crossattn_model_fn
from kdiff_trainer.xattn.get_precomputed_conds_by_ix import get_precomputed_conds_by_ix
from kdiff_trainer.xattn.crossattn_cfg_args import CrossAttnCFGArgs
from kdiff_trainer.xattn.crossattn_extra_args import CrossAttnExtraArgs
from kdiff_trainer.to_pil_images import to_pil_images

def ensure_distributed():
    if not dist.is_initialized():
        dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=64,
                   help='the batch size')
    p.add_argument('--checkpointing', action='store_true',
                   help='enable gradient checkpointing')
    p.add_argument('--clip-model', type=str, default='ViT-B/16',
                   choices=K.evaluation.CLIPFeatureExtractor.available_models(),
                   help='the CLIP model to use to evaluate')
    p.add_argument('--compile', action='store_true',
                   help='compile the model')
    p.add_argument('--config', type=str, required=True,
                   help='the configuration file')
    p.add_argument('--out-root', type=str, default='.',
                   help='outputs (checkpoints, demo samples, state, metrics) will be saved under this directory')
    p.add_argument('--output-to-subdir', action='store_true',
                   help='for outputs (checkpoints, demo samples, state, metrics): whether to use {{out_root}}/{{name}}/ subdirectories. When True: saves/loads from/to {{out_root}}/{{name}}/[{{product}}/]*, ({{product}}/ subdirectories are used for demo samples and checkpoints). When False: saves/loads from/to {{out_root}}/{{name}}_*')
    p.add_argument('--demo-every', type=int, default=500,
                   help='save a demo grid every this many steps')
    p.add_argument('--dinov2-model', type=str, default='vitl14',
                   choices=K.evaluation.DINOv2FeatureExtractor.available_models(),
                   help='the DINOv2 model to use to evaluate')
    p.add_argument('--end-step', type=int, default=None,
                   help='the step to end training at')
    p.add_argument('--evaluate-every', type=int, default=10000,
                   help='evaluate every this many steps')
    p.add_argument('--evaluate-with', type=str, default='inception',
                   choices=['inception', 'clip', 'dinov2'],
                   help='the feature extractor to use for evaluation')
    p.add_argument('--evaluate-n', type=int, default=2000,
                   help='the number of samples to draw to evaluate')
    p.add_argument('--gns', action='store_true',
                   help='measure the gradient noise scale (DDP only, disables stratified sampling)')
    p.add_argument('--grad-accum-steps', type=int, default=1,
                   help='the number of gradient accumulation steps')
    p.add_argument('--lr', type=float,
                   help='the learning rate')
    p.add_argument('--mixed-precision', type=str,
                   help='the mixed precision type')
    p.add_argument('--name', type=str, default='model',
                   help='the name of the run')
    p.add_argument('--num-workers', type=int, default=8,
                   help='the number of data loader workers')
    p.add_argument('--reset-ema', action='store_true',
                   help='reset the EMA')
    p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
    p.add_argument('--resume-inference', type=str,
                   help='the inference checkpoint to resume from')
    p.add_argument('--sample-n', type=int, default=64,
                   help='the number of images to sample for demo grids')
    p.add_argument('--save-every', type=int, default=10000,
                   help='save every this many steps')
    p.add_argument('--seed', type=int,
                   help='the random seed')
    p.add_argument('--start-method', type=str, default='spawn',
                   choices=['fork', 'forkserver', 'spawn'],
                   help='the multiprocessing start method')
    p.add_argument('--text-model-hf-cache-dir', type=str, default=None,
                   help='disk directory into which HF should download text model checkpoints')
    p.add_argument('--font', type=str, default='./kdiff_trainer/font/DejaVuSansMono.ttf',
                   help='font used for drawing demo grids (e.g. /usr/share/fonts/dejavu/DejaVuSansMono.ttf or ./kdiff_trainer/font/DejaVuSansMono.ttf). Pass empty string for ImageFont.load_default().')
    p.add_argument('--demo-title-qualifier', type=str, default=None,
                   help='Additional text to include in title printed in demo grids')
    p.add_argument('--demo-img-compress', action='store_true',
                   help='Demo image file format. False: .png; True: .jpg')
    p.add_argument('--wandb-entity', type=str,
                   help='the wandb entity name')
    p.add_argument('--wandb-group', type=str,
                   help='the wandb group name')
    p.add_argument('--wandb-project', type=str,
                   help='the wandb project name (specify this to enable wandb)')
    p.add_argument('--wandb-save-model', action='store_true',
                   help='save model to wandb')
    args = p.parse_args()

    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch._dynamo.config.automatic_dynamic_shapes = False
    except AttributeError:
        pass

    config = K.config.load_config(args.config)
    model_config = config['model']
    dataset_config = config['dataset']
    opt_config = config['optimizer']
    sched_config = config['lr_sched']
    ema_sched_config = config['ema_sched']

    # TODO: allow non-square input sizes
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps, mixed_precision=args.mixed_precision)

    ensure_distributed()
    device = accelerator.device
    unwrap = accelerator.unwrap_model
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f'World size: {accelerator.num_processes}', flush=True)
        print(f'Batch size: {args.batch_size * accelerator.num_processes}', flush=True)

    if args.seed is not None:
        seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes], generator=torch.Generator().manual_seed(args.seed))
        torch.manual_seed(seeds[accelerator.process_index])
    demo_gen = torch.Generator().manual_seed(torch.randint(-2 ** 63, 2 ** 63 - 1, ()).item())
    elapsed = 0.0

    inner_model = K.config.make_model(config)
    inner_model_ema = deepcopy(inner_model)

    if args.compile:
        inner_model.compile()
        # inner_model_ema.compile()

    if accelerator.is_main_process:
        print(f'Parameters: {K.utils.n_params(inner_model):,}')

    # If logging to wandb, initialize the run
    use_wandb = accelerator.is_main_process and args.wandb_project
    if use_wandb:
        import wandb
        log_config = vars(args)
        log_config['config'] = config
        log_config['parameters'] = K.utils.n_params(inner_model)
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group, config=log_config, save_code=True)

    lr = opt_config['lr'] if args.lr is None else args.lr
    groups = inner_model.param_groups(lr)
    inner_model, inner_model_ema = accelerator.prepare(inner_model, inner_model_ema)

    precomputed_conds: Optional[PrecomputedConds] = precompute_conds(
        accelerator=accelerator,
        dataset_config=dataset_config,
        args=args,
    ) if 'classes_to_captions' in dataset_config else None

    if opt_config['type'] == 'adamw':
        opt = optim.AdamW(groups,
                          lr=lr,
                          betas=tuple(opt_config['betas']),
                          eps=opt_config['eps'],
                          weight_decay=opt_config['weight_decay'])
    elif opt_config['type'] == 'adam8bit':
        import bitsandbytes as bnb
        opt = bnb.optim.Adam8bit(groups,
                                 lr=lr,
                                 betas=tuple(opt_config['betas']),
                                 eps=opt_config['eps'],
                                 weight_decay=opt_config['weight_decay'])
    elif opt_config['type'] == 'sgd':
        opt = optim.SGD(groups,
                        lr=lr,
                        momentum=opt_config.get('momentum', 0.),
                        nesterov=opt_config.get('nesterov', False),
                        weight_decay=opt_config.get('weight_decay', 0.))
    else:
        raise ValueError('Invalid optimizer type')
    opt_ema = optim.SGD(inner_model_ema.parameters(), lr=0.0)
    opt, opt_ema = accelerator.prepare(opt, opt_ema)

    if sched_config['type'] == 'inverse':
        sched = K.utils.InverseLR(opt,
                                  inv_gamma=sched_config['inv_gamma'],
                                  power=sched_config['power'],
                                  warmup=sched_config['warmup'])
    elif sched_config['type'] == 'exponential':
        sched = K.utils.ExponentialLR(opt,
                                      num_steps=sched_config['num_steps'],
                                      decay=sched_config['decay'],
                                      warmup=sched_config['warmup'])
    elif sched_config['type'] == 'constant':
        sched = K.utils.ConstantLRWithWarmup(opt, warmup=sched_config['warmup'])
    else:
        raise ValueError('Invalid schedule type')

    assert ema_sched_config['type'] == 'inverse'
    ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                  max_value=ema_sched_config['max_value'])
    ema_stats = {}

    uses_crossattn: bool = 'cross_attns' in model_config and model_config['cross_attns']
    if uses_crossattn:
        assert 'demo_uncond' in dataset_config
        assert dataset_config['demo_uncond'] == 'allzeros' or dataset_config['demo_uncond'] == 'emptystr'

    tf = transforms.Compose([
        transforms.Resize(size[0], interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size[0]),
        K.augmentation.KarrasAugmentationPipeline(model_config['augment_prob'], disable_all=model_config['augment_prob'] == 0),
    ])

    if dataset_config['type'] == 'imagefolder':
        train_set = K.utils.FolderOfImages(dataset_config['location'], transform=tf)
    elif dataset_config['type'] == 'imagefolder-class':
        train_set = datasets.ImageFolder(dataset_config['location'], transform=tf)
    elif dataset_config['type'] == 'cifar10':
        train_set = datasets.CIFAR10(dataset_config['location'], train=True, download=True, transform=tf)
    elif dataset_config['type'] == 'mnist':
        train_set = datasets.MNIST(dataset_config['location'], train=True, download=True, transform=tf)
    elif dataset_config['type'] == 'huggingface':
        from datasets import load_dataset
        train_set = load_dataset(dataset_config['location'])
        ds_transforms: List[DataSetTransform] = []
        if uses_crossattn:
            if 'classes_to_captions' in dataset_config:
                assert dataset_config['classes_to_captions'] == 'oxford-flowers'
                from kdiff_trainer.dataset_meta.oxford_flowers import ordinal_to_lexical
                def embed_ix_extractor(batch: BatchData) -> BatchData:
                    labels_ordinal: List[int] = batch['label']
                    # labels_ordinal is 1-indexed.
                    # the conds in text_embeds happen to be 1-indexed too, because we inserted an uncond embed at index 0
                    # had we not embedded an uncond at index 0, we would need to adapt labels_ordinal's 1-index to text_embeds' 0-index:
                    # labels_lexical_zeroix: List[int] = [ordinal_to_lexical[o]-1 for o in labels_ordinal]
                    labels_lexical: List[int] = [ordinal_to_lexical[o] for o in labels_ordinal]
                    return { 'embed_ix': labels_lexical }
                ds_transforms.append(embed_ix_extractor)
            else:
                def label_extractor(batch: BatchData) -> BatchData:
                    return { 'label': batch['label'] }
                ds_transforms.append(label_extractor)
        img_augs: DataSetTransform = partial(K.utils.hf_datasets_augs_helper, transform=tf, image_key=dataset_config['image_key'])
        ds_transforms.append(img_augs)
        multi_transform: DataSetTransform = partial(K.utils.hf_datasets_multi_transform, transforms=ds_transforms)
        train_set.set_transform(multi_transform)
        train_set = train_set['train']
    elif dataset_config['type'] == 'custom':
        location = (Path(args.config).parent / dataset_config['location']).resolve()
        spec = importlib.util.spec_from_file_location('custom_dataset', location)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        get_dataset = getattr(module, dataset_config.get('get_dataset', 'get_dataset'))
        custom_dataset_config = dataset_config.get('config', {})
        train_set = get_dataset(custom_dataset_config, transform=tf)
    else:
        raise ValueError('Invalid dataset type')

    if accelerator.is_main_process:
        try:
            print(f'Number of items in dataset: {len(train_set):,}')
        except TypeError:
            pass

    image_key = dataset_config.get('image_key', 0)
    num_classes = dataset_config.get('num_classes', 0)
    cond_dropout_rate = dataset_config.get('cond_dropout_rate', 0.1)
    class_key = dataset_config.get('class_key', 1)

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    train_dl = accelerator.prepare(train_dl)

    if use_wandb:
        wandb.watch(inner_model)
    if accelerator.num_processes == 1:
        args.gns = False
    if args.gns:
        gns_stats_hook = K.gns.DDPGradientStatsHook(inner_model)
        gns_stats = K.gns.GradientNoiseScale()
    else:
        gns_stats = None
    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    sample_density = K.config.make_sample_density(model_config)

    model = K.config.make_denoiser_wrapper(config)(inner_model)
    model_ema = K.config.make_denoiser_wrapper(config)(inner_model_ema)

    if args.output_to_subdir:
        run_root = f'{args.out_root}/{args.name}'
        state_root = metrics_root = run_root
        demo_root = f'{run_root}/demo'
        ckpt_root = f'{run_root}/ckpt'
        run_qualifier = f''
    else:
        run_root = demo_root = ckpt_root = state_root = metrics_root = args.out_root
        run_qualifier = f'{args.name}_'

    if accelerator.is_main_process:
        makedirs(run_root, exist_ok=True)
        makedirs(state_root, exist_ok=True)
        makedirs(metrics_root, exist_ok=True)
        makedirs(demo_root, exist_ok=True)
        makedirs(ckpt_root, exist_ok=True)

    state_path = Path(f'{state_root}/{run_qualifier}state.json')

    if state_path.exists() or args.resume:
        if args.resume:
            ckpt_path = args.resume
        if not args.resume:
            state = json.load(open(state_path))
            ckpt_path = f"{state_root}/{state['latest_checkpoint']}"
        if accelerator.is_main_process:
            print(f'Resuming from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        unwrap(model.inner_model).load_state_dict(ckpt['model'])
        unwrap(model_ema.inner_model).load_state_dict(ckpt['model_ema'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        ema_sched.load_state_dict(ckpt['ema_sched'])
        ema_stats = ckpt.get('ema_stats', ema_stats)
        epoch = ckpt['epoch'] + 1
        step = ckpt['step'] + 1
        if args.gns and ckpt.get('gns_stats', None) is not None:
            gns_stats.load_state_dict(ckpt['gns_stats'])
        demo_gen.set_state(ckpt['demo_gen'])
        elapsed = ckpt.get('elapsed', 0.0)

        del ckpt
    else:
        epoch = 0
        step = 0

    if args.reset_ema:
        unwrap(model.inner_model).load_state_dict(unwrap(model_ema.inner_model).state_dict())
        ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                      max_value=ema_sched_config['max_value'])
        ema_stats = {}

    if args.resume_inference:
        if accelerator.is_main_process:
            print(f'Loading {args.resume_inference}...')
        ckpt = safetorch.load_file(args.resume_inference)
        unwrap(model.inner_model).load_state_dict(ckpt)
        unwrap(model_ema.inner_model).load_state_dict(ckpt)
        del ckpt

    evaluate_enabled = args.evaluate_every > 0 and args.evaluate_n > 0
    if evaluate_enabled:
        if args.evaluate_with == 'inception':
            extractor = K.evaluation.InceptionV3FeatureExtractor(device=device)
        elif args.evaluate_with == 'clip':
            extractor = K.evaluation.CLIPFeatureExtractor(args.clip_model, device=device)
        elif args.evaluate_with == 'dinov2':
            extractor = K.evaluation.DINOv2FeatureExtractor(args.dinov2_model, device=device)
        else:
            raise ValueError('Invalid evaluation feature extractor')
        train_iter = iter(train_dl)
        if accelerator.is_main_process:
            print('Computing features for reals...')
        reals_features = K.evaluation.compute_features(accelerator, lambda x: next(train_iter)[image_key][1], extractor, args.evaluate_n, args.batch_size)
        if accelerator.is_main_process:
            metrics_log = K.utils.CSVLogger(f'{metrics_root}/{run_qualifier}metrics.csv', ['step', 'time', 'loss', 'fid', 'kid'])
        del train_iter

    cfg_scale = 1.

    def make_cfg_model_fn(model):
        def cfg_model_fn(x, sigma, class_cond):
            x_in = torch.cat([x, x])
            sigma_in = torch.cat([sigma, sigma])
            class_uncond = torch.full_like(class_cond, num_classes)
            class_cond_in = torch.cat([class_uncond, class_cond])
            out = model(x_in, sigma_in, class_cond=class_cond_in)
            out_uncond, out_cond = out.chunk(2)
            return out_uncond + (out_cond - out_uncond) * cfg_scale
        if cfg_scale != 1:
            return cfg_model_fn
        return model

    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    def demo(captioner: GridCaptioner):
        if accelerator.is_main_process:
            tqdm.write('Sampling...')
        with FSDP.summon_full_params(model_ema):
            pass
        n_per_proc = math.ceil(args.sample_n / accelerator.num_processes)
        x = torch.randn([accelerator.num_processes, n_per_proc, model_config['input_channels'], size[0], size[1]], generator=demo_gen).to(device)
        dist.broadcast(x, 0)
        x = x[accelerator.process_index] * sigma_max
        model_fn, extra_args = model_ema, {}
        caption_ix: Optional[LongTensor] = None
        if uses_crossattn:
            assert precomputed_conds is not None
            cfg_args: CrossAttnCFGArgs = get_precomputed_cond_cfg_args(
                accelerator=accelerator,
                precomputed_conds=precomputed_conds,
                uncond_type=dataset_config['demo_uncond'],
                n_per_proc=n_per_proc,
                distribute=True,
                rng=demo_gen,
            )
            caption_ix = cfg_args.caption_ix
            extra_args = {**extra_args, **cfg_args.sampling_extra_args}
            model_fn = make_cfg_crossattn_model_fn(model_ema, masked_uncond=cfg_args.masked_uncond, cfg_scale=cfg_scale)
        elif num_classes:
            class_cond = torch.randint(0, num_classes, [accelerator.num_processes, n_per_proc], generator=demo_gen).to(device)
            dist.broadcast(class_cond, 0)
            extra_args['class_cond'] = class_cond[accelerator.process_index]
            model_fn = make_cfg_model_fn(model_ema)
        sigmas = K.sampling.get_sigmas_karras(50, sigma_min, sigma_max, rho=7., device=device)
        x_0: FloatTensor = K.sampling.sample_dpmpp_2m_sde(model_fn, x, sigmas, extra_args=extra_args, eta=0.0, solver_type='heun', disable=not accelerator.is_main_process)
        x_0 = accelerator.gather(x_0)[:args.sample_n]
        if accelerator.is_main_process:
            if uses_crossattn:
                assert caption_ix is not None
                pils: List[Image.Image] = to_pil_images(x_0)
                captions: List[str] = [precomputed_conds.class_captions[caption_ix_.item()] for caption_ix_ in caption_ix.flatten().cpu()]
                title = f'[step {step}] {args.name} {args.config}'
                if args.demo_title_qualifier:
                    title += f' {args.demo_title_qualifier}'
                grid_pil: Image.Image = captioner.__call__(
                    imgs=pils,
                    captions=captions,
                    title=title,
                )
            else:
                grid = utils.make_grid(x_0, nrow=math.ceil(args.sample_n ** 0.5), padding=0)
                grid_pil: Image.Image = K.utils.to_pil_image(grid)
            save_kwargs = { 'subsampling': 0, 'quality': 95 } if args.demo_img_compress else {}
            fext = 'jpg' if args.demo_img_compress else 'png'
            filename = f'{demo_root}/{run_qualifier}{step:08}.{fext}'
            grid_pil.save(filename, **save_kwargs)

            if use_wandb:
                wandb.log({'demo_grid': wandb.Image(filename)}, step=step)

    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    def evaluate():
        if not evaluate_enabled:
            return
        if accelerator.is_main_process:
            tqdm.write('Evaluating...')
        with FSDP.summon_full_params(model_ema):
            pass
        sigmas = K.sampling.get_sigmas_karras(50, sigma_min, sigma_max, rho=7., device=device)
        def sample_fn(n):
            x = torch.randn([n, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
            model_fn, extra_args = model_ema, {}
            if uses_crossattn:
                assert precomputed_conds is not None
                cfg_args: CrossAttnCFGArgs = get_precomputed_cond_cfg_args(
                    accelerator=accelerator,
                    precomputed_conds=precomputed_conds,
                    uncond_type=dataset_config['eval_uncond'],
                    n_per_proc=n,
                    distribute=False,
                    rng=None,
                )
                extra_args = {**extra_args, **cfg_args.sampling_extra_args}
                model_fn = make_cfg_crossattn_model_fn(model_ema, masked_uncond=cfg_args.masked_uncond, cfg_scale=cfg_scale)
            elif num_classes:
                extra_args['class_cond'] = torch.randint(0, num_classes, [n], device=device)
                model_fn = make_cfg_model_fn(model_ema)
            x_0 = K.sampling.sample_dpmpp_2m_sde(model_fn, x, sigmas, extra_args=extra_args, eta=0.0, solver_type='heun', disable=True)
            return x_0
        fakes_features = K.evaluation.compute_features(accelerator, sample_fn, extractor, args.evaluate_n, args.batch_size)
        if accelerator.is_main_process:
            fid = K.evaluation.fid(fakes_features, reals_features)
            kid = K.evaluation.kid(fakes_features, reals_features)
            print(f'FID: {fid.item():g}, KID: {kid.item():g}')
            if accelerator.is_main_process:
                metrics_log.write(step, elapsed, ema_stats['loss'], fid.item(), kid.item())
            if use_wandb:
                wandb.log({'FID': fid.item(), 'KID': kid.item()}, step=step)

    def save():
        accelerator.wait_for_everyone()
        filename = f'{ckpt_root}/{run_qualifier}{step:08}.pth'
        if accelerator.is_main_process:
            tqdm.write(f'Saving to {filename}...')

        with (
            FSDP.summon_full_params(model.inner_model, rank0_only=True, offload_to_cpu=True, writeback=False),
            FSDP.summon_full_params(model_ema.inner_model, rank0_only=True, offload_to_cpu=True, writeback=False),
        ):
            inner_model = unwrap(model.inner_model)
            inner_model_ema = unwrap(model_ema.inner_model)
            if accelerator.is_main_process:
                obj = {
                    'config': config,
                    'model': inner_model.state_dict(),
                    'model_ema': inner_model_ema.state_dict(),
                    'opt': opt.state_dict(),
                    'sched': sched.state_dict(),
                    'ema_sched': ema_sched.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'gns_stats': gns_stats.state_dict() if gns_stats is not None else None,
                    'ema_stats': ema_stats,
                    'demo_gen': demo_gen.get_state(),
                    'elapsed': elapsed,
                }
                accelerator.save(obj, filename)
                state_obj = {'latest_checkpoint': relpath(filename, state_root)}
                json.dump(state_obj, open(state_path, 'w'))
                if args.wandb_save_model and use_wandb:
                    wandb.save(filename)

    losses_since_last_print = []

    try:
        while True:
            for batch in tqdm(train_dl, smoothing=0.1, disable=not accelerator.is_main_process):
                if device.type == 'cuda':
                    start_timer = torch.cuda.Event(enable_timing=True)
                    end_timer = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start_timer.record()
                else:
                    start_timer = time.time()

                with accelerator.accumulate(model):
                    reals, _, aug_cond = batch[image_key]
                    class_cond, extra_args = None, {}
                    if precomputed_conds is not None:
                        xattn_extra_args: CrossAttnExtraArgs = get_precomputed_conds_by_ix(
                            device=accelerator.device,
                            precomputed_conds=precomputed_conds,
                            embed_ix=batch['embed_ix'],
                            cond_dropout_rate=cond_dropout_rate,
                            allzeros_uncond_rate=dataset_config['allzeros_uncond_rate'],
                        )
                        extra_args = {**extra_args, **xattn_extra_args}
                    elif num_classes:
                        class_cond = batch[class_key]
                        drop = torch.rand(class_cond.shape, device=class_cond.device)
                        class_cond.masked_fill_(drop < cond_dropout_rate, num_classes)
                        extra_args['class_cond'] = class_cond
                    noise = torch.randn_like(reals)
                    with K.utils.enable_stratified_accelerate(accelerator, disable=args.gns):
                        sigma = sample_density([reals.shape[0]], device=device)
                    with K.models.checkpointing(args.checkpointing):
                        losses = model.loss(reals, noise, sigma, aug_cond=aug_cond, **extra_args)
                    loss = accelerator.gather(losses).mean().item()
                    losses_since_last_print.append(loss)
                    accelerator.backward(losses.mean())
                    if args.gns:
                        sq_norm_small_batch, sq_norm_large_batch = gns_stats_hook.get_stats()
                        gns_stats.update(sq_norm_small_batch, sq_norm_large_batch, reals.shape[0], reals.shape[0] * accelerator.num_processes)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.)
                    opt.step()
                    sched.step()
                    opt.zero_grad()

                    ema_decay = ema_sched.get_value()
                    K.utils.ema_update_dict(ema_stats, {'loss': loss}, ema_decay ** (1 / args.grad_accum_steps))
                    if accelerator.sync_gradients:
                        K.utils.ema_update(model, model_ema, ema_decay)
                        ema_sched.step()

                if device.type == 'cuda':
                    end_timer.record()
                    torch.cuda.synchronize()
                    elapsed += start_timer.elapsed_time(end_timer) / 1000
                else:
                    elapsed += time.time() - start_timer

                if step % 25 == 0:
                    loss_disp = sum(losses_since_last_print) / len(losses_since_last_print)
                    losses_since_last_print.clear()
                    avg_loss = ema_stats['loss']
                    if accelerator.is_main_process:
                        if args.gns:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}, gns: {gns_stats.get_gns():g}')
                        else:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}')

                if use_wandb:
                    log_dict = {
                        'epoch': epoch,
                        'loss': loss,
                        'lr': sched.get_last_lr()[0],
                        'ema_decay': ema_decay,
                    }
                    if args.gns:
                        log_dict['gradient_noise_scale'] = gns_stats.get_gns()
                    wandb.log(log_dict, step=step)

                step += 1

                pad = 8
                cell_pad = BBox(top=pad, left=pad, bottom=pad, right=pad)
                # abusing bottom padding to simulate a margin-bottom
                title_pad = BBox(top=pad, left=pad, bottom=pad*3, right=pad)

                cell_font: ImageFont = ImageFont.truetype(args.font, 25) if args.font else ImageFont.load_default()
                title_font: ImageFont = ImageFont.truetype(args.font, 50) if args.font else ImageFont.load_default()

                cols: int = math.ceil(args.sample_n ** .5)
                # TODO: are h and w the right way around?
                samp_h, samp_w = model_config['input_size']
                cell_type: Typesetting = make_typesetting(
                    x_wrap_px=samp_w,
                    font=cell_font,
                    padding=cell_pad,
                )
                title_type: Typesetting = make_typesetting(
                    x_wrap_px=samp_w*cols,
                    font=title_font,
                    padding=title_pad,
                )

                captioner: GridCaptioner = make_grid_captioner(
                    cell_type=cell_type,
                    cols=cols,
                    samp_w=samp_w,
                    samp_h=samp_h,
                    title_type=title_type,
                )

                if step % args.demo_every == 0:
                    demo(captioner)

                if evaluate_enabled and step > 0 and step % args.evaluate_every == 0:
                    evaluate()

                if step == args.end_step or (step > 0 and step % args.save_every == 0):
                    save()

                if step == args.end_step:
                    if accelerator.is_main_process:
                        tqdm.write('Done!')
                    return

            epoch += 1
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
