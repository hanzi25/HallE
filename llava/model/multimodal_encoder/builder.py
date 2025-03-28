import os
from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    
    if vision_tower == "openai/clip-vit-large-patch14-336":
        vision_tower = "/raid_sdd/zzy/model/clip_vit_large_patch14_336"

    is_absolute_path_exists = os.path.exists(vision_tower)    
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
