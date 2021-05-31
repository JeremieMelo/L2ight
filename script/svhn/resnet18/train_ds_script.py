import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from torchpack.utils.config import configs

root = "log/svhn/resnet18/ds"
script = 'train_learn.py'
config_file = 'config/svhn/resnet18/ds/learn.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    fbs, cs, ds, id = args
    with open(os.path.join(root, f'btopk-exp-fbs-{fbs}-none-cs-{cs}-ss-0-smd-ds-{ds}_run-{id}.log'), 'w') as wfid:
        exp = [f"--sparse.bp_data_sparsity={ds}",
                "--sparse.bp_data_alg=smd",
               "--sparse.bp_spatial_sparsity=0",
               f"--sparse.bp_column_sparsity={cs}",
               "--sparse.bp_input_norm=none",
               "--sparse.bp_input_sparsify_first_conv=0",
               f"--sparse.bp_feedback_weight_sparsity={fbs}",
               "--sparse.bp_feedback_alg=topk",
               "--sparse.bp_feedback_norm=exp",
               f"--sparse.random_state={41+id}",
                f"--run.random_state={41+id}"
               ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    # tasks = [(0.6, 0.6, 0.5, 1)] # 23922  07:39 PM 04/27
    tasks = [(0.5, 0.5, 0.5, 1)] # 5640  10:42 AM 05/06
    # tasks = [(0.5, 0.5, 0.5, 2)] # 6147  09:04 PM 04/30
    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
