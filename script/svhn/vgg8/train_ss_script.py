import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "log/cifar10/vgg8/ss"
script = 'train_learn.py'
config_file = 'config/cifar10/vgg8/ss/learn.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    fbs, fbs_norm, ss = args
    with open(os.path.join(root, f'norm-{fbs_norm}_fbs-{fbs}_norm-none_cs-0_ss-{ss}.log'), 'w') as wfid:
        exp = [
             f"--sparse.bp_feedback_weight_sparsity={fbs}",
               "--sparse.bp_feedback_alg=topk",
               f"--sparse.bp_feedback_norm={fbs_norm}",
            f"--sparse.bp_spatial_sparsity={ss}",
               "--sparse.bp_column_sparsity=0",
               "--sparse.bp_input_norm=none",
               "--sparse.bp_input_sparsify_first_conv=0"]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    #         fbs   fbs_norm ss
    # tasks = [[0.6, "exp", 0.4]] # 16366  3:57 AM 4/25
    # ## for RAD
    tasks = [[0.6, "exp", 0.4]] # # w/o first conv
    # # w/o first conv
    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
