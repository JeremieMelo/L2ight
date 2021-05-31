import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from torchpack.utils.config import configs

root = "log/cifar10/vgg8/cs"
script = 'train_learn.py'
config_file = 'config/cifar10/vgg8/cs/learn.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    fbs, ss, cs, first, input_norm, id = args
    with open(os.path.join(root, f'fbs-{fbs}_norm-{input_norm}_first-{first}_ss-{ss}_cs-{cs}_run-{id}.log'), 'w') as wfid:
        exp = [
            f"--sparse.bp_feedback_weight_sparsity={fbs}",
            "--sparse.bp_feedback_alg=topk",
            "--sparse.bp_feedback_norm=exp",
            f"--sparse.bp_spatial_sparsity={ss}",
            f"--sparse.bp_column_sparsity={cs}",
            f"--sparse.bp_input_sparsify_first_conv={first}",
            f"--sparse.bp_input_norm={input_norm}",
            f"--sparse.random_state={41+id}",
            f"--run.random_state={41+id}"]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    # fbs, ss, cs, first, input_norm, id
    tasks = [[0.6, 0, 0.5, 0, "none", 1]] # 17798  3:58 AM 4/25
    tasks = [[0.6, 0, 0.4, 0, "none", 1]] # 20541  04:55 AM 4/25
    tasks = [[0.6, 0, 0.6, 0, "none", 1]] # 20772  04:56 AM 4/25
    tasks = [[0.6, 0.4, 0.4, 0, "none", 1]] # 20670  04:56 AM 4/25
    tasks = [[0.6, 0.4, 0.4, 1, "none", 1]] # 1366  09:53 PM 4/25
    tasks = [[0.6, 0, 0.6, 1, "none", 1]] # 2377  10:03 PM 4/25 w/o ss, first conv=1
    tasks = [[0.6, 0, 0.6, 0, "exp", 1]] # 1684  02:23 AM 4/26 w/o ss, first conv=0, exp norm
    # tasks = [[0.6, 0, 0.6, 0, "var"]] # 2377  10:03 PM 4/25 w/o ss, first conv=0, var norm
    tasks = [[0.6, 0, 0.6, 0, "none", 2]] # 22880  05:26 PM 4/30
    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
