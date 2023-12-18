CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
# Model Constants
IGNORE_INDEX = -100
X_TOKEN_INDEX = {'IMAGE': -200, 'VIDEO': -201, 'BEV': -202, 'MAP': -203, 'LIDAR': -204}
X_INDEX_TOKEN = {v: k for k, v in X_TOKEN_INDEX.items()}
# IMAGE_TOKEN_INDEX = -200
DEFAULT_X_TOKEN = {'IMAGE': "<image>", 'VIDEO': "<video>", 'BEV': "<bev>", 'MAP': "<map>"}
# DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_X_PATCH_TOKEN = {'IMAGE': "<im_patch>", 'VIDEO': "<vi_patch>", 'BEV': "<bev_patch>", 'MAP': "<map_patch>"}
# DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_X_START_TOKEN = {'IMAGE': "<im_start>", 'VIDEO': "<vi_start>", 'BEV': "<bev_start>", 'MAP': "<map_start>"}
# DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_X_END_TOKEN = {'IMAGE': "<im_end>", 'VIDEO': "<vi_end>", 'BEV': "<bev_end>", 'MAP': "<map_end>"}
# DEFAULT_IM_END_TOKEN = "<im_end>"
