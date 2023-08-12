import os
import sys
import time

PREFIX = "your path to lra"

batches = {
    "cifar": [200],
    "imdb": 64,
    "listops": 128,
    "pathfinder": 128,
    "pathfinderx": 64,
    "aan": 64,
}

gpus = {
    "cifar": 2,
    "imdb": 4,
    "listops": 2,
    "pathfinder": 4,
    "pathfinderx": 8,
    "aan": 2,
}

d_model_dict = {
    "cifar": 128,
    "imdb": 128,
    "listops": 128,
    "pathfinder": 128,
    "pathfinderx": 64,
    "aan": 128,
}

n_layers_dict = {
    "cifar": 12,
    "imdb": [4],
    "listops": 4,
    "pathfinder": 4,
    "pathfinderx": 6,
    "aan": 4,
}

expand_ratio_ffn_dict = {
    "cifar": 2,
    "imdb": 4,
    "listops": 4,
    "pathfinder": 1,
    "pathfinderx": 2.5,
    "aan": 4,
}

norm_dict = {
    "cifar": "synbatch",
    "imdb": "synbatch",
    "listops": "synbatch",
    "pathfinder": "synbatch",
    "pathfinderx": "synbatch",
    "aan": "synbatch",
}

lr_dict = {
    "cifar": [0.001],
    "imdb": [0.001],
    "listops": 0.0001,
    "pathfinder": [0.0005,],
    "pathfinderx": [0.0001],
    "aan": [0.001],
}

prenorm_dict = {
    "cifar": True,
    "imdb": True,
    "listops": True,
    "pathfinder": True,
    "pathfinderx": True,
    "aan": True,
}

head_dict = {
    "cifar": 4,
    "imdb": 8,
    "listops": 8,
    "pathfinder": 8,
    "pathfinderx": 8,
    "aan": 4,
}


archs = ["transnormer"]

tasks = ["aan"]
tasks = ["imdb"]
tasks = ["listops"]
tasks = ["cifar"]
tasks = ["pathfinder"]

# t1
model_config = [[False, "elu"]]

# # t2
# model_config = [[True, "1+elu"]]


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
        batches[task],
        lr_dict[task],
        n_layers_dict[task],
        d_model_dict[task],
        norm_dict[task],
        prenorm_dict[task],
        model_config,
        expand_ratio_ffn_dict[task],
        head_dict[task],
    )
    print(pars)
    print(task)
    print(len(pars))
    time.sleep(10)
    for (
        arch,
        total_batch,
        lr,
        n_layers,
        d_model,
        norm,
        prenorm,
        config,
        expand_ratio_ffn,
        num_heads,
    ) in pars:
        use_softmax, act_fun = config
        print(use_softmax, act_fun)
        if task == "imdb":
            seq_len = 4096
            gpu = gpus[task]
            batch = total_batch // gpu
            workers = gpu * 20
            for i in range(1):
                print("imdb lr: ", lr)
                time.sleep(10)
                pid = os.fork()
                if pid == 0:
                    os.system(
                        f"sh {PREFIX}/train_lra.sh {task} {arch} {total_batch} {lr} {n_layers} {d_model} {norm} {prenorm} {use_softmax} {act_fun} {expand_ratio_ffn} {num_heads} {gpu}"
                    )
                    sys.exit(0)
        elif task == "cifar":
            seq_len = 1024
            gpu = gpus[task]
            batch = total_batch // gpu
            workers = gpu * 20
            print("cifar lr: ", lr)
            time.sleep(10)
            pid = os.fork()
            if pid == 0:
                os.system(
                    f"sh {PREFIX}/train_lra.sh {task} {arch} {total_batch} {lr} {n_layers} {d_model} {norm} {prenorm} {use_softmax} {act_fun} {expand_ratio_ffn} {num_heads} {gpu}"
                )
                sys.exit(0)
        elif task == "listops":
            seq_len = 2048
            gpu = gpus[task]
            batch = total_batch // gpu
            workers = gpu * 20
            print("listops lr: ", lr)
            time.sleep(10)
            pid = os.fork()
            if pid == 0:
                os.system(
                    f"sh {PREFIX}/train_lra.sh {task} {arch} {total_batch} {lr} {n_layers} {d_model} {norm} {prenorm} {use_softmax} {act_fun} {expand_ratio_ffn} {num_heads} {gpu}"
                )
                sys.exit(0)
        elif task == "pathfinder":
            seq_len = 1024
            gpu = gpus[task]
            batch = total_batch // gpu
            workers = gpu * 20
            print("pathfinder lr: ", lr)
            time.sleep(10)
            pid = os.fork()
            if pid == 0:
                os.system(
                    f"sh {PREFIX}/train_lra.sh {task} {arch} {total_batch} {lr} {n_layers} {d_model} {norm} {prenorm} {use_softmax} {act_fun} {expand_ratio_ffn} {num_heads} {gpu}"
                )
                sys.exit(0)
        elif task == "pathfinderx":
            seq_len = 128 * 128
            gpu = gpus[task]
            batch = total_batch // gpu
            workers = gpu * 20
            print("pathfinderx lr: ", lr)
            time.sleep(10)
            pid = os.fork()
            if pid == 0:
                os.system(
                    f"sh {PREFIX}/train_lra.sh {task} {arch} {total_batch} {lr} {n_layers} {d_model} {norm} {prenorm} {use_softmax} {act_fun} {expand_ratio_ffn} {num_heads} {gpu}"
                )
                sys.exit(0)
        elif task == "aan":
            seq_len = 4000
            gpu = gpus[task]
            batch = total_batch // gpu
            workers = gpu * 20
            print("aan lr: ", lr)
            time.sleep(10)
            pid = os.fork()
            if pid == 0:
                os.system(
                    f"sh {PREFIX}/train_lra.sh {task} {arch} {total_batch} {lr} {n_layers} {d_model} {norm} {prenorm} {use_softmax} {act_fun} {expand_ratio_ffn} {num_heads} {gpu}"
                )
                sys.exit(0)
