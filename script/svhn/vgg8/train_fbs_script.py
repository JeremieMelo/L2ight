import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from torchpack.utils.config import configs

root = "log/svhn/vgg8/fbs"
script = 'train_learn.py'
config_file = 'config/svhn/vgg8/fbs/learn.yml'
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
    # 4798  11:28 PM 04/24 s=0
    # 7243  11:52 PM 04/24 s=0.6
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    # s = [(0.6, 1)]
    # tasks = [(0.6, 2), (0, 2)] # 25848  05:24 PM 04/30
    tasks = [(0, 1)] # 19317  05:24 PM 05/05
    with Pool(2) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
