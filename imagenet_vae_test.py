from kdiff_trainer.dataset.get_dataset import get_dataset
import k_diffusion as K

from pathlib import Path

#Just manually defining config right now..

config_path = "configs/config_guided_diffusion_imagenet_latent.jsonc"
config = K.config.load_config(config_path, use_json5=config_path.endswith('.jsonc'))
print(config)


data = get_dataset(config["dataset"], config_dir=Path(config_path).parent, 
            uses_crossattn=False, tf=None, class_captions=None)


for item in data:
    print(item)
    