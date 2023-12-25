import papermill as pm
from datetime import datetime, timedelta
from train import get_save_folder
from copy import deepcopy
import json
from config import Config


import click

@click.command()
@click.option("-d", "--dry-run", is_flag=True, help="Dry run")
def run(dry_run):
    # all params
    default_params = {
        "BATCH_SIZE" : 8,
        "N_EPOCHS" : 10,
        "SUBMIT" : True,
        # model
        "BACKBONE" : 'resnet18',
        "PRETRAINED": False,
        "FPN" : False,
        # aug
        "CUT_AUG" : None,
        "COLOR_AUG" : None,
        "FLIP_AUG" : False,
        "SIDE_EXT": 4,
        # train
        "SWITCH_LOSS_EPOCH": 5, # this does not do anything
        "MASK_LOSS_TYPE" : "default",
        #ONLY useful with bce
        "LABEL_SMOOTHING" : 0.0,
        "TQDM_DISABLED" : True,
        "SCHEDULER": "Step",
        "LEARNING_RATE": 0.001,
        "MILESTONE": [0.6, 0.9],
        # img and mask utils
        "DISTANCE_THRESH_CLEAR": 2,
        "IMG_WIDTH": 1600,
        "IMG_HEIGHT": 704,
        "MODEL_SCALE": 8,
        "OPTIMIZE_XY": True,
        # CTIME_STR
        # PAPERMILL_INPUT_PATH
        # PAPERMILL_OUTPUT_PATH
    }

    params = [
        {},
    ]

    combined_params = []
    for param in params:
        combined_params.append(deepcopy(default_params))
        combined_params[-1].update(param)

    total_seconds = 0
    base_reso = 1024 * 320
    for param in combined_params:
        if "IMG_HEIGHT" not in param or param["IMG_HEIGHT"] is None:
            param["IMG_HEIGHT"] = (param["IMG_WIDTH"] // 16 * 5 + 32 // 2) // 32 * 32
        Config.validate(param)
        total_seconds += param["IMG_WIDTH"] * param["IMG_HEIGHT"] / base_reso * (3*60+50) * param["N_EPOCHS"]

    print("params: ", json.dumps(combined_params, indent=4))

    # estimate total run time
    diff = timedelta(seconds=total_seconds)
    print("total run time: ", diff)
    print("estimated end time: ", datetime.now() + diff)

    if dry_run:
        return
    for param in combined_params:
        extra_config = ["fpn" if param["FPN"] else None, 
                        "cut" if param["CUT_AUG"] else None, 
                        "color" if param["COLOR_AUG"] else None,
                        param["MASK_LOSS_TYPE"] if param["MASK_LOSS_TYPE"] != "default" else None,
                        param["SCHEDULER"],
        ]
        extra_config_str = "".join([f"-{x}" for x in extra_config if x is not None])
        param.update({"CTIME_STR": datetime.now().strftime("%Y-%m-%d-%H-%M-%S")})
        param.update({"PAPERMILL_INPUT_PATH": "centernet-train.ipynb"})
        param.update({"PAPERMILL_OUTPUT_PATH": f"{param['CTIME_STR']}-centernet-train-{param['BACKBONE']}-epoch-{param['N_EPOCHS']}{extra_config_str}.ipynb"})
        SAVE_FOLDER = get_save_folder(param["PAPERMILL_OUTPUT_PATH"], param["CTIME_STR"])
        pm.execute_notebook(
            param["PAPERMILL_INPUT_PATH"],
            param["PAPERMILL_OUTPUT_PATH"],
            parameters=param,
            log_output=True
        )

if __name__ == "__main__":
    run()