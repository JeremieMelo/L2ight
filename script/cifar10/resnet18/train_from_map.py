'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:45:45
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-24 16:45:45
'''

import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "log/cifar10/vgg8/cs"
script = 'train_learn.py'
config_file = 'config/cifar10/vgg8/cs/learn.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    fbs, ss, cs, first, input_norm, ds, acc, checkpoint = args
    with open(os.path.join(root, f'ds-{ds}_fbs-{fbs}_norm-{input_norm}_first-{first}_ss-{ss}_cs-{cs}_mapacc-{acc}.log'), 'w') as wfid:
        exp = [
            "--optimizer.lr=0.001",
            f"--sparse.bp_data_sparsity={ds}",
            "--sparse.bp_data_alg=smd",
            f"--sparse.bp_feedback_weight_sparsity={fbs}",
            "--sparse.bp_feedback_alg=topk",
            "--sparse.bp_feedback_norm=exp",
            f"--sparse.bp_spatial_sparsity={ss}",
            f"--sparse.bp_column_sparsity={cs}",
            f"--sparse.bp_input_sparsify_first_conv={first}",
            f"--sparse.bp_input_norm={input_norm}",
            f"--checkpoint.resume=1",
            f"--checkpoint.restore_checkpoint={checkpoint}",
            "--run.n_epochs=50",
            f"--checkpoint.model_comment=mapacc-{acc}"]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    args = [[0.6, 0, 0.6, 0, "none", 0.5]]
    checkpoints = [[acc,os.path.join("./checkpoint/cifar10/vgg8/pm", i)] for acc, i in [
        [59.62, "SparseBP_MZI_VGG8_wb-8_ib-32_icalg-zcd_icadapt-0_icbest-1__acc-59.62_epoch-20.pt"],
        [66.23,"SparseBP_MZI_VGG8_wb-8_ib-32_icalg-zcd_icadapt-0_icbest-1__acc-66.23_epoch-40.pt"],
        [69.89,"SparseBP_MZI_VGG8_wb-8_ib-32_icalg-zcd_icadapt-0_icbest-1__acc-69.89_epoch-60.pt"],
        [76.06, "SparseBP_MZI_VGG8_wb-8_ib-32_icalg-zcd_icadapt-0_icbest-1__acc-76.06_epoch-80.pt"],
        [83.19,"SparseBP_MZI_VGG8_wb-8_ib-32_icalg-zcd_icadapt-0_icbest-1__acc-83.19_epoch-120.pt"],
        [85.19,"SparseBP_MZI_VGG8_wb-8_ib-32_icalg-zcd_icadapt-0_icbest-1__acc-85.19_epoch-150.pt"],
        [89.13,"SparseBP_MZI_VGG8_wb-8_ib-32_icalg-zcd_icadapt-0_icbest-1__acc-89.13_epoch-300.pt"]
        ]]
    tasks = [args[0]+i for i in checkpoints]
    with Pool(1) as p:
        p.map(task_launcher, tasks[3:4])
    logger.info(f"Exp: {configs.run.experiment} Done.")
