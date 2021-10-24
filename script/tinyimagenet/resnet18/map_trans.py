'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:42:34
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-24 16:42:34
'''

import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "log/cifar10/resnet18/trans"
script = 'train_map.py'
config_file = 'config/cifar10/resnet18/trans/pm.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    epoch = args
    with open(os.path.join(root, f'pretrain-cifar100_ic-zcd-b-400_pm-ztp-b-{epoch}.log'), 'w') as wfid:
        exp = [
            f"--ic.alg=zcd",
            f"--ic.best_record=1",
            f"--ic.adaptive=0",
            f"--pm.alg=ztp",
            f"--pm.best_record=1",
            f"--pm.adaptive=0",
            f"--run.cali_n_epochs=400",
            f"--run.map_n_epochs={epoch}",
            f"--checkpoint.model_comment=ic-400",
            f"--checkpoint.no_linear=0",
            f"--checkpoint.resume=1",
            f"--checkpoint.restore_checkpoint=./checkpoint/cifar100/resnet18/pretrain/SparseBP_MZI_ResNet18_wb-32_ib-32__acc-68.94_epoch-187.pt"]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    tasks3 = [400] # ZTP DONE
    with Pool(1) as p:
        p.map(task_launcher, tasks3)
    logger.info(f"Exp: {configs.run.experiment} Done.")
