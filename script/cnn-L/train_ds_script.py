import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "log/fmnist/cnn3/ds"
script = 'train_learn.py'
config_file = 'config/fmnist/cnn3/ds/learn.yml'
configs.load(config_file, recursive=True)


def task_launcher(s: float):
    pres = ['python3',
            script,
            config_file
            ]
    with open(os.path.join(root, f'btopk-exp-fbs-0.6-exp-ss-0-cs-0.6-smd-ds-{s}.log'), 'w') as wfid:
        exp = [f"--sparse.bp_data_sparsity={s}",
                "--sparse.bp_data_alg=smd",
               "--sparse.bp_spatial_sparsity=0",
               "--sparse.bp_column_sparsity=0.6",
               "--sparse.bp_input_norm=exp",
               "--sparse.bp_input_sparsify_first_conv=1",
               "--sparse.bp_feedback_weight_sparsity=0.6",
               "--sparse.bp_feedback_alg=topk",
               "--sparse.bp_feedback_norm=exp"
               ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    #
    s = [0.8, 0.9] # DONE
    s = [0.7] #
    with Pool(1) as p:
        p.map(task_launcher, s)
    logger.info(f"Exp: {configs.run.experiment} Done.")
