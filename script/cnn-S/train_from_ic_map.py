
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "log/mnist/cnn3/ds"
script = 'train_learn.py'
config_file = 'config/mnist/cnn3/ds/learn.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    fbs, ss, cs, first, input_norm, ds, acc, checkpoint = args
    with open(os.path.join(root, f'ds-{ds}_fbs-{fbs}_norm-{input_norm}_first-{first}_ss-{ss}_cs-{cs}_ic-400_mapacc-{acc}.log'), 'w') as wfid:
        exp = [
            "--optimizer.lr=0.0001",
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
            f"--sl.noisy_identity=1",
            f"--checkpoint.restore_checkpoint={checkpoint}",
            "--run.n_epochs=10",
            f"--checkpoint.model_comment=ic-400_mapacc-{acc}"]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    # w/o ss, first conv=0, exp norm
    # checkpoints = [[acc,os.path.join("./checkpoint/cifar10/vgg8/pm", i)] for acc, i in [
    #     [59.62, "SparseBP_MZI_VGG8_wb-8_ib-32_icalg-zcd_icadapt-0_icbest-1_ic-400_acc-59.62_epoch-20.pt"],
    #     [66.23,"SparseBP_MZI_VGG8_wb-8_ib-32_icalg-zcd_icadapt-0_icbest-1_ic-400_acc-66.23_epoch-40.pt"],
    #     [69.89,"SparseBP_MZI_VGG8_wb-8_ib-32_icalg-zcd_icadapt-0_icbest-1_ic-400_acc-69.89_epoch-60.pt"],
    #     [76.06, "SparseBP_MZI_VGG8_wb-8_ib-32_icalg-zcd_icadapt-0_icbest-1_ic-400_acc-76.06_epoch-80.pt"],
    #     [83.19,"SparseBP_MZI_VGG8_wb-8_ib-32_icalg-zcd_icadapt-0_icbest-1_ic-400_acc-83.19_epoch-120.pt"],
    #     [85.02,"SparseBP_MZI_VGG8_wb-8_ib-32_icalg-zcd_icadapt-0_icbest-1_ic-400_acc-85.02_epoch-200.pt"],
    #     [86.35,"SparseBP_MZI_VGG8_wb-8_ib-32_icalg-zcd_icadapt-0_icbest-1_ic-400_acc-86.35_epoch-300.pt"]
    #     ]]
    tasks = [(0.6, 0, 0.6, 0,"none", 0.5, 94.66, "./checkpoint/mnist/cnn3/pm/SparseBP_MZI_CNN_wb-8_ib-32_icalg-zcd_icadapt-0_icbest-1__acc-94.66_epoch-300.pt")]
    # DONE
    #
    # tasks = [args[0]+i for i in checkpoints]
    # tasks1 = tasks[:3] # 15043
    # tasks2 = tasks[3:6] # 28117
    # tasks3 = tasks[6:] # 22773
    # tasks4 = tasks[-2:-1] # 29085 ead14
    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
