from mmengine import MODELS, Registry
from mmengine.dataset import default_collate
from mmengine.dist import get_dist_info
from mmengine.registry import build_from_cfg



def build_sampler(cfg, default_args):
    return build_from_cfg(cfg, MODELS, default_args)
