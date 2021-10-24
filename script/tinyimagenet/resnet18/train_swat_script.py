
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "log/tinyimagenet/resnet18/ss"
script = 'train_learn.py'
config_file = 'config/tinyimagenet/resnet18/ss/learn.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    fws, fws_alg, fws_norm, ss, ss_norm, id = args
    with open(os.path.join(root, f'norm-{fws_norm}_fws-{fws_alg}-{fws}_norm-{ss_norm}_ss-{ss}_run-{id}.log'), 'w') as wfid:
        exp = [
            f"--sparse.bp_forward_weight_sparsity={fws}",
            f"--sparse.bp_feedback_weight_sparsity=0",
            f"--sparse.bp_feedback_alg={fws_alg}",
            f"--sparse.bp_feedback_norm={fws_norm}",
            f"--sparse.bp_spatial_sparsity={ss}",
            f"--sparse.bp_column_sparsity=0",
            f"--sparse.bp_input_norm={ss_norm}",
            f"--sparse.bp_input_sparsify_first_conv=0",
            f"--sparse.random_state={41+id}",
            f"--run.random_state={41+id}"]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    #         fws   fws_alg  fws_norm ss ss_norm
    #
    # killed
    tasks = [[0.3, "uniform", "exp", 0.5, "none", 2]] #

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
