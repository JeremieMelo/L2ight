import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "log/fmnist/cnn3/fbs"
script = 'train_learn.py'
config_file = 'config/fmnist/cnn3/fbs/learn.yml'
configs.load(config_file, recursive=True)


def task_launcher(s: float):
    pres = ['python3',
            script,
            config_file
            ]
    with open(os.path.join(root, f'gtopk_norm-none_fbs-{s}.log'), 'w') as wfid:
        exp = [f"--sparse.bp_feedback_weight_sparsity={s}",
               "--sparse.bp_feedback_alg=gtopk",
               "--sparse.bp_feedback_norm=none",
               "--checkpoint.model_comment=gtopk"]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    # PID 21133 . 06:02 AM
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    s = [0.2, 0.4, 0.6, 0.8, 0.9]
    with Pool(5) as p:
        p.map(task_launcher, s)
    logger.info(f"Exp: {configs.run.experiment} Done.")
