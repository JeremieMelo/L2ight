
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from torchpack.utils.config import configs

root = "log/svhn/vgg8/ds"
script = 'train_learn.py'
config_file = 'config/svhn/vgg8/ds/learn.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    fbs, ss, cs, first, input_norm, ds, id, acc, checkpoint = args
    with open(os.path.join(root, f'ds-{ds}_fbs-{fbs}_norm-{input_norm}_first-{first}_ss-{ss}_cs-{cs}_ic-400_mapacc-{acc}_run-{id}.log'), 'w') as wfid:
        exp = [
            "--optimizer.lr=0.0002",
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
            "--run.n_epochs=20",
            f"--checkpoint.model_comment=ic-400_mapacc-{acc}",
            f"--sparse.random_state={41+id}",
            f"--run.random_state={41+id}"]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    # fbs, ss, cs, first, input_norm, ds, id, acc, checkpoint
    tasks = [[0.6, 0, 0.6, 0, "none", 0.5, 1, 92.57, "./checkpoint/svhn/vgg8/pm/SparseBP_MZI_VGG8_wb-8_ib-32_icalg-zcd_icadapt-0_icbest-1_ic-400_acc-92.57_epoch-300.pt"]] # 9052 03:59 PM 05/06

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
