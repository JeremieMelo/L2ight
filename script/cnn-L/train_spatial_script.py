import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "log/fmnist/cnn3/ss"
script = 'train_learn.py'
config_file = 'config/fmnist/cnn3/ss/learn.yml'
configs.load(config_file, recursive=True)


def task_launcher(s: float):
    pres = ['python3',
            script,
            config_file
            ]
    with open(os.path.join(root, f'norm-none_cs-0_ss-{s}.log'), 'w') as wfid:
        exp = [f"--sparse.bp_spatial_sparsity={s}",
               "--sparse.bp_column_sparsity=0",
               "--sparse.bp_input_norm=none"]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    # 22541  06:58 PM
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    s = [0.2, 0.6, 0.9]
    with Pool(3) as p:
        p.map(task_launcher, s)
    logger.info(f"Exp: train_spatial Done.")
