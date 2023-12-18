import os
from .clip_encoder import CLIPVisionTower
from .bev_encoder import BEVVisionTower



def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_bev_tower(vision_tower_cfg, **kwargs):
    bev_tower = getattr(vision_tower_cfg, 'mm_bev_tower', getattr(vision_tower_cfg, 'bev_tower', None))
    is_absolute_path_exists = os.path.exists(bev_tower)
    if is_absolute_path_exists:
        return BEVVisionTower(bev_tower, args=None, **kwargs)

    raise ValueError(f'Unknown vision tower: {bev_tower}')
