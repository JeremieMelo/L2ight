'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:42:28
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-24 16:42:28
'''

import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "log/tinyimagenet/resnet18/pm"
script = 'train_map.py'
config_file = 'config/tinyimagenet/resnet18/pm/pm.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    epoch, map_alg, id = args
    with open(os.path.join(root, f'ic-zcd-b-400_pm-{map_alg}-b-a-{epoch}_run-{id}.log'), 'w') as wfid:
        exp = [
            f"--ic.alg=zcd",
            f"--ic.best_record=1",
            f"--ic.adaptive=0",
            f"--pm.alg={map_alg}",
            f"--pm.best_record=1",
            f"--pm.adaptive=1",
            f"--run.cali_n_epochs=400",
            f"--run.map_n_epochs={epoch}",
            f"--checkpoint.model_comment=ic-400"]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    # tasks1 = [20,40,60,80]
    # tasks2 = [120,200]
    tasks3 = [(300, "ztp", 1)] #
    tasks3 = [(300, "zcd", 1)] #
    with Pool(1) as p:
        p.map(task_launcher, tasks3)
    logger.info(f"Exp: {configs.run.experiment} Done.")
