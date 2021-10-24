import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "log/cifar100/resnet18/fbs"
script = 'train_learn.py'
config_file = 'config/cifar100/resnet18/fbs/learn.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    fbs, id = args
    with open(os.path.join(root, f'btopk-exp-fbs-{fbs}_run-{id}.log'), 'w') as wfid:
        exp = [
               f"--sparse.bp_feedback_weight_sparsity={fbs}",
               "--sparse.bp_feedback_alg=topk",
               "--sparse.bp_feedback_norm=exp",
               f"--sparse.random_state={41+id}",
               f"--run.random_state={41+id}"
               ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    # s=0
    # s=0.6
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    # 0.6 not DONE wrong profiler
    #
    s = [(0, 1)] #
    #
    with Pool(1) as p:
        p.map(task_launcher, s)
    logger.info(f"Exp: {configs.run.experiment} Done.")
