'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:41:50
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-24 16:41:50
'''
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "log/tinyimagenet/vgg8/pretrain"
script = 'train_pretrain.py'
config_file = 'config/tinyimagenet/vgg8/pretrain.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    data_name, n_class = args
    with open(os.path.join(root, f'pretrain_{data_name}.log'), 'w') as wfid:
        exp = [
            f"--dataset.name={data_name}",
            f"--dataset.n_class={n_class}"
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    tasks = [("tinyimagenet", 200)]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
