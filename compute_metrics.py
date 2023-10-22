import accelerate
import argparse
import k_diffusion as K
import importlib.util
from pathlib import Path
import torch
from torch import distributed as dist, multiprocessing as mp
from torch.utils import data
from torchvision import datasets, transforms
from functools import partial
from typing import Dict, Optional

def ensure_distributed():
    if not dist.is_initialized():
        dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())

def get_train_set(dataset_config: Dict, config_path: str, tf: Optional[transforms.Compose] = None):
    match dataset_config['type']:
        case 'imagefolder':
            train_set = K.utils.FolderOfImages(dataset_config['location'], transform=tf)
        case 'imagefolder-class':
            train_set = datasets.ImageFolder(dataset_config['location'], transform=tf)
        case 'cifar10':
            train_set = datasets.CIFAR10(dataset_config['location'], train=True, download=True, transform=tf)
        case 'mnist':
            train_set = datasets.MNIST(dataset_config['location'], train=True, download=True, transform=tf)
        case 'huggingface':
            from datasets import load_dataset
            train_set = load_dataset(dataset_config['location'])
            train_set.set_transform(partial(K.utils.hf_datasets_augs_helper, transform=tf, image_key=dataset_config['image_key']))
            train_set = train_set['train']
        case 'custom':
            location = (Path(config_path).parent / dataset_config['location']).resolve()
            spec = importlib.util.spec_from_file_location('custom_dataset', location)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            get_dataset = getattr(module, dataset_config.get('get_dataset', 'get_dataset'))
            custom_dataset_config = dataset_config.get('config', {})
            train_set = get_dataset(custom_dataset_config, transform=tf)
        case _:
            raise ValueError(f"Invalid dataset type '{dataset_config['type']}'")
    return train_set

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=64,
                   help='the batch size')
    p.add_argument('--config-pred', type=str, required=True,
                   help='configuration file detailing a dataset of predictions from a model')
    p.add_argument('--config-target', type=str, required=True,
                   help='configuration file detailing a dataset of ground-truth examples')
    p.add_argument('--evaluate-n', type=int, default=2000,
                   help='the number of samples to draw to evaluate')
    p.add_argument('--evaluate-with', type=str, default='inception',
                   choices=['inception', 'clip', 'dinov2'],
                   help='the feature extractor to use for evaluation')
    # p.add_argument('--transform-interp', type=transforms.InterpolationMode,
    #                default=transforms.InterpolationMode.BICUBIC,
    #                help='which interpolation mode used for resizing ground-truth images')
    p.add_argument('--clip-model', type=str, default='ViT-B/16',
                   choices=K.evaluation.CLIPFeatureExtractor.available_models(),
                   help='the CLIP model to use to evaluate')
    p.add_argument('--dinov2-model', type=str, default='vitl14',
                   choices=K.evaluation.DINOv2FeatureExtractor.available_models(),
                   help='the DINOv2 model to use to evaluate')
    p.add_argument('--mixed-precision', type=str,
                   choices=['no', 'fp16', 'bf16', 'fp8'],
                   help='the mixed precision type')
    args = p.parse_args()

    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch._dynamo.config.automatic_dynamic_shapes = False
    except AttributeError:
        pass
    
    accelerator = accelerate.Accelerator(mixed_precision=args.mixed_precision)
    ensure_distributed()
    device = accelerator.device

    config_pred = K.config.load_config(args.config_pred, use_json5=args.config.endswith('.jsonc'))
    model_config = config_pred['model']
    pred_dataset_config = config_pred['dataset']

    # TODO: allow non-square input sizes
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    config_target = K.config.load_config(args.config_target, use_json5=args.config.endswith('.jsonc'))
    target_dataset_config = config_target['dataset']

    pred_image_key, target_image_key = (config.get('image_key', 0) for config in (pred_dataset_config, target_dataset_config))
    num_classes = pred_dataset_config.get('num_classes', 0)
    cond_dropout_rate = pred_dataset_config.get('cond_dropout_rate', 0.1)
    class_key = pred_dataset_config.get('class_key', 1)

    tf = transforms.Compose([
        transforms.Resize(size[0], interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size[0]),
    ])

    pred_train_set, target_train_set = (
        get_train_set(dataset_config, config_path, tf=tf) for dataset_config, config_path in (
            (pred_dataset_config, args.config_pred),
            (target_dataset_config, args.config_target),
        )
    )
    target_train_set = get_train_set(dataset_config=target_dataset_config, config_path=args.config_target, tf=tf)

    if accelerator.is_main_process:
        try:
            for set_name, train_set in zip(('pred', 'target'), (pred_train_set, target_train_set)):
                print(f'Number of items in {set_name} dataset: {len(train_set):,}')
        except TypeError:
            pass

    pred_train_dl, target_train_dl = (data.DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, persistent_workers=True, pin_memory=True) for train_set in (pred_train_set, target_train_set))
    pred_train_dl, target_train_dl = accelerator.prepare(pred_train_dl, target_train_dl)

    match args.evaluate_with:
        case 'inception':
            extractor = K.evaluation.InceptionV3FeatureExtractor(device=device)
        case 'clip':
            extractor = K.evaluation.CLIPFeatureExtractor(args.clip_model, device=device)
        case 'dinov2':
            extractor = K.evaluation.DINOv2FeatureExtractor(args.dinov2_model, device=device)
        case _:
            raise ValueError(f"Invalid evaluation feature extractor '{args.evaluate_with}'")
    
    pred_train_iter, target_train_iter = (iter(dl) for dl in (pred_train_dl, target_train_dl))
    if accelerator.is_main_process:
        for source_name, iter, image_key in zip(('pred', 'target'), (pred_train_iter, target_train_iter), (pred_image_key, target_image_key)):
            print(f'Computing features for {source_name}...')
            features = K.evaluation.compute_features(accelerator, lambda _: next(iter)[image_key][1], extractor, args.evaluate_n, args.batch_size)
    del train_iter



if __name__ == '__main__':
    main()
