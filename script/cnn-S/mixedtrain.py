import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from torchpack.utils.config import configs

root = "log/mnist/cnn3/mixedtrain"
script = 'train_zo_learn.py'
config_file = 'config/mnist/cnn3/mixedtrain/learn.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    param_sparsity, grad_sparsity = args
    with open(os.path.join(root, f'mixedtrain_ps-{param_sparsity}_gs-{grad_sparsity}.log'), 'w') as wfid:
        exp = [
            f"--optimizer.param_sparsity={param_sparsity}",
            f"--optimizer.grad_sparsity={grad_sparsity}",
            f"--run.n_epochs=20"
               ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [(0.85, 0.9)] # 14590  01:42 AM 04/27
    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
