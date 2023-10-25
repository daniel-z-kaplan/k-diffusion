from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ClassCaptions:
    # for display purposes in demo grids
    demo_class_captions: List[str]
    # for cross-attn embedding
    embed_class_captions: List[str]
    uncond_class_ix: int
    # we deliberately model this mapping as a List rather than a Callable, to make it easier to pickle (it's used in a closure that the dataloader copies to spawned subprocesses)
    # TODO: looks like that wasn't enough. class_ix_extractor only works in fork mode for now.
    dataset_label_to_canonical_label: Optional[List[int]]

def get_class_captions(
   classes_to_captions: str,
) -> ClassCaptions:
    if classes_to_captions == 'oxford-flowers':
        from kdiff_trainer.dataset_meta.oxford_flowers import flower_classes, nelorth_to_fatima
        labels_excl_uncond: List[str] = flower_classes
        dataset_label_to_canonical_label: List[int] = nelorth_to_fatima
    elif classes_to_captions == 'imagenet-1k':
        from kdiff_trainer.dataset_meta.imagenet_1k import class_labels
        labels_excl_uncond: List[str] = class_labels
        dataset_label_to_canonical_label: Optional[List[int]] = None
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