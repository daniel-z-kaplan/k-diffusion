from dataclasses import dataclass
from typing import List, Optional, Callable

@dataclass
class ClassCaptions:
    # for display purposes in demo grids
    demo_class_captions: List[str]
    # for cross-attn embedding
    embed_class_captions: List[str]
    uncond_class_ix: int
    dataset_label_to_canonical_label: Optional[Callable[[int], int]]

def get_class_captions(
   classes_to_captions: str,
) -> ClassCaptions:
    if classes_to_captions == 'oxford-flowers':
        from kdiff_trainer.dataset_meta.oxford_flowers import flower_classes, nelorth_to_fatima
        labels_excl_uncond: List[str] = flower_classes
        def dataset_label_to_canonical_label(label_nelorth: int) -> int:
            return nelorth_to_fatima[label_nelorth]
    elif classes_to_captions == 'imagenet-1k':
        from kdiff_trainer.dataset_meta.imagenet_1k import class_labels
        labels_excl_uncond: List[str] = class_labels
        dataset_label_to_canonical_label: Optional[Callable[[int], int]]
    else:
        raise ValueError(f"Never heard of classes_to_captions '{classes_to_captions}'")
    demo_class_captions: List[str] = [*labels_excl_uncond, '<UNCOND>']
    embed_class_captions: List[str] = [*labels_excl_uncond, '']
    uncond_class_ix: int = len(demo_class_captions)-1
    return ClassCaptions(
        demo_class_captions=demo_class_captions,
        embed_class_captions=embed_class_captions,
        uncond_class_ix=uncond_class_ix,
        dataset_label_to_canonical_label=dataset_label_to_canonical_label,
    )