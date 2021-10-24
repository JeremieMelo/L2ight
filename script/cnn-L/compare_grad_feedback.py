
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "log/fmnist/cnn3/fbs_cg"
script = 'compare_gradient.py'
config_file = 'config/fmnist/cnn3/fbs/cg.yml'
configs.load(config_file, recursive=True)


def task_launcher(s: float):
    pres = ['python3',
            script,
            config_file
            ]
    with open(os.path.join(root, f'topk_norm-var_fbs-{s}.log'), 'w') as wfid:
        exp = [f"--sparse.bp_feedback_weight_sparsity={s}",
               "--sparse.bp_feedback_alg=topk",
               "--sparse.bp_feedback_norm=var"]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':

    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    # s = [0.2, 0.4, 0.6, 0.8, 0.9]
    # s1 = [0.2, 0.4] # 24738
    # s2 = [0.6, 0.8] # 25653
    # s3 = [0.9] # 11663
    s = [0.6] # exp norm 5724  07:15PM
    s = [0.6] # var 5986  07:15 PM
    with Pool(1) as p:
        p.map(task_launcher, s)
    logger.info(f"Exp: {configs.run.experiment} Done.")
