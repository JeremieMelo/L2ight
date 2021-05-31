
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from torchpack.utils.config import configs

root = "log/cifar10/vgg8/trans"
script = 'train_map.py'
config_file = 'config/cifar10/vgg8/trans/pm.yml'
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
            f"--checkpoint.no_linear=0"]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    # tasks1 = [20,40,60,80]
    # tasks2 = [120,200]
    # tasks3 = [300] # ZCD DONE
    tasks3 = [400] # 7485  12:33 AM 04/28 ZTP DONE
    with Pool(1) as p:
        p.map(task_launcher, tasks3)
    logger.info(f"Exp: {configs.run.experiment} Done.")