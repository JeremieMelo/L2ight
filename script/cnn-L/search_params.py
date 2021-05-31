import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from torchpack.utils.config import configs

root = "log/fmnist/cnn2/search"
script = 'train_learn.py'
config_file = 'config/fmnist/cnn2/search/learn.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    fs, fbs, ss, cs = args
    pres = ['python3',
            script,
            config_file
            ]
    with open(os.path.join(root, f'fcfbs-0_fs-{fs}_norm-exp_fbs-{fbs}_ss-{ss}_cs-{cs}.log'), 'w') as wfid:
        exp = [f"--sparse.bp_feedback_weight_sparsity={fbs}",
               "--sparse.bp_feedback_alg=topk",
               "--sparse.bp_feedback_norm=exp",
               f"--sparse.bp_spatial_sparsity={ss}",
               f"--sparse.bp_column_sparsity={cs}",
               f"--sparse.bp_input_sparsity={fs}",
               f"--sparse.bp_input_norm=none",
               f"--run.n_epochs=20"]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    #  6259
    #  22883
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
            # fs   fbs   ss   cs
    # tasks = [(0.5, 0.5, 0.5, 0.5),
    #          (0.6, 0.6, 0.6, 0.6),
    #          (0.7, 0.7, 0.7, 0.7),
    #          (0.8, 0.8, 0.8, 0.8)] ## course search
    ### coordinate search
    tasks = []
    for fs in [0]:#[0.6, 0.7, 0.8, 0.9]:
        for fbs in [0.6,0.7,0.8,0.9]:
            for ss in [0.6]:#[0.6,0.7,0.8,0.9]:
                for cs in [0.9]:#[0.6,0.7,0.8,0.9]:
                    tasks.append((fs,fbs,ss,cs))

    # tasks_1 = tasks[:len(tasks)//3] #
    # tasks_2 = tasks[len(tasks)//3:-len(tasks)//3] #
    # tasks_3 = tasks[-len(tasks)//3:] #

    with Pool(2) as p:
        p.map(task_launcher, tasks[2:])
    logger.info(f"Exp: {configs.run.experiment} Done.")
