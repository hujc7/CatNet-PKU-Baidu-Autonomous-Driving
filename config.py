from collections import defaultdict
import numpy as np
import inspect
import copy

camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)

camera_matrix_inv = np.linalg.inv(camera_matrix)

# DISTANCE_THRESH_CLEAR = 2

# IMG_WIDTH = 1024
# IMG_HEIGHT = IMG_WIDTH // 16 * 5
# MODEL_SCALE = 8

IMG_SHAPE = (2710, 3384)

PATH = 'data/'

class Config:
    PARAMS = {
        "BATCH_SIZE": int,
        "N_EPOCHS": int,
        "BACKBONE": ["resnet18", "resnet50", "resnet101", "unet"],
        "SUBMIT": bool,
        "PAPERMILL_INPUT_PATH": None,
        "PAPERMILL_OUTPUT_PATH": None,
        "TQDM_DISABLED": bool,
        # AUG
        "FPN": bool,
        "CUT_AUG": np.linspace(0, 100, 101).tolist() + [None],
        "COLOR_AUG": np.linspace(0, 100, 101).tolist() + [None],
        # LOSS
        "MASK_LOSS_TYPE": ["default", "focal", "bce"],
        "LABEL_SMOOTHING": float,
        "CTIME_STR": str,
        # SCHEDULER
        "SCHEDULER": ["Step", "MultiStep"],
        "LEARNING_RATE": float,
        "MILESTONE": list,
        "IMG_WIDTH": int,
        "IMG_HEIGHT": int,
        "MODEL_SCALE": int,
        "DISTANCE_THRESH_CLEAR": int,
        "SWITCH_LOSS_EPOCH": int,
        "PRETRAINED": bool,
        "OPTIMIZE_XY": bool,
        "FLIP_AUG": bool,
        "SIDE_EXT": int,
    }
    def __init__(self, default_params):
        self.params = defaultdict(None)
        self.update(default_params)

    def update(self, params):
        for k, v in params.items():
            # check if key is registered
            if k not in self.PARAMS:
                raise ValueError(f"param {k} not in PARAMS")
            
            # check if type match or within range
            if self.PARAMS[k] is not None:
                if inspect.isclass(self.PARAMS[k]):
                    if not isinstance(v, self.PARAMS[k]):
                        raise ValueError(f"param {k} type {type(v)} not match {self.PARAMS[k]}")
                elif not v in self.PARAMS[k]:
                    raise ValueError(f"param {k} value {v} not in {self.PARAMS[k]}")
            self.params[k] = copy.deepcopy(v)
    
    def update_and_copy(self, params):
        new_config = copy.deepcopy(self)
        new_config.update(params)
        return new_config
    
    @staticmethod
    def validate(params):
        for k, v in params.items():
            # check if key is registered
            if k not in Config.PARAMS:
                raise ValueError(f"param {k} not in PARAMS")
            
            # check if type match or within range
            if Config.PARAMS[k] is not None:
                if inspect.isclass(Config.PARAMS[k]):
                    if not isinstance(v, Config.PARAMS[k]):
                        raise ValueError(f"param {k} type {type(v)} not match {Config.PARAMS[k]}")
                elif not v in Config.PARAMS[k]:
                    raise ValueError(f"param {k} value {v} not in {Config.PARAMS[k]}")
        
        if "IMG_WIDTH" in params and "MODEL_SCALE" in params and params["IMG_WIDTH"] % (params["MODEL_SCALE"] * 4) != 0:
            raise ValueError(f"IMG_WIDTH {params['IMG_WIDTH']} % MODEL_SCALE {params['MODEL_SCALE']} != 0")
        if "IMG_WIDTH" in params and "MODEL_SCALE" in params and params["IMG_HEIGHT"] % (params["MODEL_SCALE"] * 4) != 0:
            raise ValueError(f"IMG_HEIGHT {params['IMG_HEIGHT']} % MODEL_SCALE {params['MODEL_SCALE']} != 0")