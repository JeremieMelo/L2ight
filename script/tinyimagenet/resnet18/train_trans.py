import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from torchpack.utils.config import configs

root = "log/tinyimagenet/resnet18/trans"
script = 'train_learn.py'
config_file = 'config/tinyimagenet/resnet18/trans/learn.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    dataset, n_class, img_height, img_width, resume, pretrain, id = args
    with open(os.path.join(root, f'{dataset}_resume-{resume}_pretrain-{pretrain}_btopk-exp-fbs-0.5-none-cs-0.5-ss-0-smd-ds-0.5_run-{id}.log'), 'w') as wfid:
        exp = [f"--run.experiment={dataset}_resnet18_learn_trans",
               f"--dataset.name={dataset}",
               f"--dataset.n_class={n_class}",
               f"--dataset.img_height={img_height}",
               f"--dataset.img_width={img_width}",
               f"--dataset.shuffle=1",
               f"--checkpoint.checkpoint_dir=tinyimagenet/resnet18/trans/{dataset}",
               f"--sparse.bp_data_sparsity=0.5",
               "--sparse.bp_data_alg=smd",
               "--sparse.bp_spatial_sparsity=0",
               "--sparse.bp_column_sparsity=0.5",
               "--sparse.bp_input_norm=none",
               "--sparse.bp_input_sparsify_first_conv=0",
               "--sparse.bp_feedback_weight_sparsity=0.5",
               "--sparse.bp_feedback_alg=topk",
               "--sparse.bp_feedback_norm=exp",
               f"--checkpoint.resume={resume}",
               f"--checkpoint.imagenet_pretrain={pretrain}",
               "--checkpoint.no_linear=1",
               "--sl.noisy_identity=0"
               ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    # tasks = [("cifar10", 10, 32, 32, 1, 0, 1), #  20746 05:26 PM 05/10
    #          ("cifar100", 100, 32, 32, 1, 0, 1), #  898 05:25 PM 05/10
    #          ("dogs", 120, 224, 224, 0, 0, 1), # 4696  01:37 AM 05/11
    #          ("cars", 196, 224, 224, 0, 0, 1), # 14657  01:37 AM 05/11
    #          ]
    tasks = [("cifar10", 10, 32, 32, 1, 0, 2), #  2089 01:39 AM 05/12
             ("cifar100", 100, 32, 32, 1, 0, 2), #  17544 01:57 AM 05/12
             ("dogs", 120, 224, 224, 1, 0, 2), # 17582  01:44 AM 05/12
             ("cars", 196, 224, 224, 1, 0, 2), # 31892  01:45 AM 05/12
             ]
    task1 = tasks[0:1]
    task2 = tasks[1:2]
    task3 = tasks[2:3]
    task4 = tasks[3:4]
    with Pool(1) as p:
        p.map(task_launcher, task2)
    logger.info(f"Exp: {configs.run.experiment} Done.")
