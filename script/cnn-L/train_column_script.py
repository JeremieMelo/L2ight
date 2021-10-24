import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "log/fmnist/cnn3/cs"
script = 'train_learn.py'
config_file = 'config/fmnist/cnn3/cs/learn.yml'
configs.load(config_file, recursive=True)


def task_launcher(s: float):
    pres = ['python3',
            script,
            config_file
            ]
    with open(os.path.join(root, f'norm-none_ss-0_cs-{s}.log'), 'w') as wfid:
        exp = ["--sparse.bp_spatial_sparsity=0",
               f"--sparse.bp_column_sparsity={s}",
               "--sparse.bp_input_norm=none"]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    s = [0.2, 0.6, 0.9]
    s1 = [0.2, 0.6] # 749  07:03 PM
    s2 = [0.9] # 3671  07:03 PM
    with Pool(1) as p:
        p.map(task_launcher, s2)
    logger.info(f"Exp: {configs.run.experiment} Done.")
