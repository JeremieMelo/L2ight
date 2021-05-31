import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from torchpack.utils.config import configs

root = "log/tinyimagenet/resnet18/pretrain"
script = 'train_pretrain.py'
config_file = 'config/tinyimagenet/resnet18/pretrain.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    data_name, n_class = args
    with open(os.path.join(root, f'pretrain_{data_name}_imagenet-pretrain-1.log'), 'w') as wfid:
        exp = [
            f"--dataset.name={data_name}",
            f"--dataset.n_class={n_class}",
            f"--checkpoint.imagenet_pretrain=1",
            f"--run.n_epochs=100",
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    tasks = [("tinyimagenet", 200)] # 3536  04:37 AM 05/07

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
