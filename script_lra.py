import os
import sys
import time

# change
use_decay = True
PREFIX = "your path to lra"

batches = {
    "cifar": 50,
    "imdb": 32,
    "listops": 128,
    "pathfinder": 128,
    "pathfinderx": 16,
    "aan": 64,
}

gpus = {
    "cifar": 1,
    "imdb": 2,
    "listops": 2,
    "pathfinder": 1,
    "pathfinderx": 4,
    "aan": 1,
}

d_model_dict = {
    "cifar": 512,
    "imdb": [128],
    "listops": 32,
    "pathfinder": [128],
    "pathfinderx": 64,
    "aan": 64,
}

n_layers_dict = {
    "cifar": [6],
    "imdb": [4],
    "listops": [6],
    "pathfinder": 6,
    "pathfinderx": 6,
    "aan": 2,
}

norm_dict = {
    "cifar": "batch",
    "imdb": "synbatch",
    "listops": "synbatch",
    "pathfinder": "batch",
    "pathfinderx": "synbatch",
    "aan": "batch",
}

lr_dict = {
    "cifar": [3e-3], 
    "imdb": [0.005],
    "listops": [0.0005],
    "pathfinder": [2e-3],
    "pathfinderx": [0.00075],
    "aan": [0.005], 
}

wd_dict = {
    "cifar": 0,
    "imdb": [0],
    "listops": [0.01],
    "pathfinder": 0,
    "pathfinderx": 0,
    "aan": [0,], 
}

dropout_dict = {
    "cifar": [0],
    "imdb": 0.1,
    "listops": [0],
    "pathfinder": 0,
    "pathfinderx": 0,
    "aan": [0],
}

prenorm_dict = {
    "cifar": [True],
    "imdb": True,
    "listops": True,
    "pathfinder": [True,],
    "pathfinderx": True,
    "aan": True,
}

warmup_steps_dict = {
    "cifar": 30000,
    "imdb": [10000],
    "listops": [5000],
    "pathfinder": [50000],
    "pathfinderx": [150000],
    "aan": [312],
}

training_steps_dict = {
    "cifar": 50000,
    "imdb": 50000,
    "listops": [50000],
    "pathfinder": [500000,],
    "pathfinderx": 500000,
    "aan": [50000],
}

expand_ratio_glu_dict = {
    "cifar": 1,
    "imdb": [1],
    "listops": [1],
    "pathfinder": [1],
    "pathfinderx": 1,
    "aan": [2],
}

param_share_dict = {
    "cifar": False,
    "imdb": False,
    "listops": False,
    "pathfinder": False,
    "pathfinderx": False,
    "aan": False,
}

training_epochs_dict = {
    "cifar": [100],
    "imdb": 32,
    "listops": [50],
    "pathfinder": [200],
    "pathfinderx": 100,
    "aan": [20],
}

use_lower_bound_dict = {
    "cifar": True,
    "imdb": True,
    "listops": True,
    "pathfinder": True,
    "pathfinderx": True,
    "aan": True,
}

# # for ablation
# use_lower_bound_dict = {
#     "cifar": False,
#     "imdb": False,
#     "listops": False,
#     "pathfinder": False,
#     "pathfinderx": False,
#     "aan": False,
# }

causal_dict = {
    "cifar": False,
    "imdb": False,
    "listops": [False],
    "pathfinder": False,
    "pathfinderx": False,
    "aan": False,
}

use_real_dict = {
    "cifar": False,
    "imdb": False,
    "listops": False,
    "pathfinder": False,
    "pathfinderx": False,
    "aan": False,
}

encoder_dict = {
    "cifar": "position",
    "imdb": "position",
    "listops": "position",
    "pathfinder": "id",
    "pathfinderx": "id",
    "aan": "position",
}

tasks = ["aan", "imdb", "listops", "cifar", "pathfinder", "pathfinderx"]
# archs = ["hgru1d"] for "aan", "imdb", "listops"
# archs = ["hgru2d"] for "cifar", "pathfinder", "pathfinderx"

archs = ["hgru1d"]

def to_iter(*args):
    n = len(args)
    new_args = []
    for i in range(n):
        if not isinstance(args[i], list):
            arg = [args[i]]
        else:
            arg = args[i]
        new_args.append(arg)

    return helper(*new_args)


def helper(*args):
    n = len(args)
    if n == 1:
        res = [[arg] for arg in args[0]]
        return res
    else:
        arr = helper(*args[1:])
        res = []
        for par in args[0]:
            for data in arr:
                res.append([par] + list(data))
        return res


for i, task in enumerate(tasks):
    pars = to_iter(
        archs,
        n_layers_dict[task],
        d_model_dict[task],
        batches[task],
        norm_dict[task],
        lr_dict[task],
        wd_dict[task],
        dropout_dict[task],
        prenorm_dict[task],
        warmup_steps_dict[task],
        training_steps_dict[task],
        expand_ratio_glu_dict[task],
        param_share_dict[task],
        training_epochs_dict[task],
        use_lower_bound_dict[task],
        causal_dict[task],
        use_real_dict[task],
        encoder_dict[task],
    )
    print(pars)
    print(task)
    print(len(pars))
    time.sleep(10)
    for (
        arch,
        n_layers,
        d_model,
        total_batch,
        norm,
        lr,
        wd,
        dropout,
        prenorm,
        warmup_steps,
        training_steps,
        expand_ratio_glu,
        param_share,
        training_epochs,
        use_lower_bound,
        causal,
        use_real,
        encoder,
    ) in pars:
        gpu = gpus[task]
        batch = total_batch // gpu
        workers = gpu * 20
        print("imdb lr: ", lr)
        if task == "cifar":
            # time.sleep(10)
            pid = os.fork()
            if pid == 0:
                os.system(
                    f"sh {PREFIX}/train_lra_image.sh {task} {arch} {batch} {n_layers} {d_model} {norm} {lr} {wd} {gpu} {workers} {dropout} {prenorm} {warmup_steps} {training_steps} {expand_ratio_glu} {param_share} {training_epochs} {use_lower_bound} {causal} {use_real} {encoder}"
                )
                sys.exit(0)
        else:
            # time.sleep(10)
            pid = os.fork()
            if pid == 0:
                os.system(
                    f"sh {PREFIX}/train_lra.sh {task} {arch} {batch} {n_layers} {d_model} {norm} {lr} {wd} {gpu} {workers} {dropout} {prenorm} {warmup_steps} {training_steps} {expand_ratio_glu} {param_share} {training_epochs} {use_lower_bound} {causal} {use_real} {encoder}"
                )
                sys.exit(0)