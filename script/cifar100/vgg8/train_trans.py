import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from torchpack.utils.config import configs

root = "log/cifar10/vgg8/trans"
script = 'train_learn.py'
config_file = 'config/cifar10/vgg8/trans/learn.yml'
configs.load(config_file, recursive=True)


def task_launcher(s: float):
    pres = ['python3',
            script,
            config_file
            ]
    with open(os.path.join(root, f'btopk-exp-fbs-0.6-none-cs-0.6-ss-0-smd-ds-{s}.log'), 'w') as wfid:
        exp = [f"--sparse.bp_data_sparsity={s}",
                "--sparse.bp_data_alg=smd",
               "--sparse.bp_spatial_sparsity=0",
               "--sparse.bp_column_sparsity=0.6",
               "--sparse.bp_input_norm=none",
               "--sparse.bp_input_sparsify_first_conv=0",
               "--sparse.bp_feedback_weight_sparsity=0.6",
               "--sparse.bp_feedback_alg=topk",
               "--sparse.bp_feedback_norm=exp",
               "--checkpoint.resume=1",
               "--checkpoint.no_linear=1",
               "--sl.noisy_identity=1"
               ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    s = [0.5] # 11960  01:06 AM 04/27
    with Pool(1) as p:
        p.map(task_launcher, s)
    logger.info(f"Exp: {configs.run.experiment} Done.")
