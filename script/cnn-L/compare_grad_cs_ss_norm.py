
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

root = "log/fmnist/cnn3/cs_ss_cg"
script = 'compare_gradient.py'
config_file = 'config/fmnist/cnn3/cs_ss/cg.yml'
configs.load(config_file, recursive=True)

### compare none, var, exp normalization in conv feature sampling, assume same spatial and column sparsity
def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    ss_sparsity, cs_sparsity, norm = args
    with open(os.path.join(root, f'norm-{norm}_ss-{ss_sparsity}_cs-{cs_sparsity}.log'), 'w') as wfid:
        exp = [f"--sparse.bp_spatial_sparsity={ss_sparsity}",
               f"--sparse.bp_column_sparsity={cs_sparsity}",
               f"--sparse.bp_input_norm={norm}"]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    #
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    tasks = [(0.6, 0.6, "none"), (0.6,0.6,"exp"), (0.6,0.6,"var")]
    #
    tasks = [(0, 0.6,"var")] #
    tasks = [(0, 0.6,"none")] #
    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
