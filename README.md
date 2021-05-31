# L2ight: Enabling On-Chip Learning for Optical Neural Networks via Efficient in-situ Subspace Optimization

# Dependencies
* Python >= 3.6
* Python libraries listed in `requirements.txt`
* NVIDIA GPUs and CUDA >= 10.2

# Structures
* core/
    * models/
        * layers/
            * custom_conv2d and custom_linear layers
            * utils.py: sampler and profiler
        * sparse_bp_\*.py: model definition
        * sparse_bp_base.py: base model definition; identity calibration and mapping codes.
    * optimizer/: mixedtrain and flops optimizers
    * builder.py: build training utilities
* onnlib/: third party
* pyutils/: third party to provide utility and CUDA-accelerated PTC simulation
* script/: contains experiment scripts
* train_pretrain.py, train_map.py, train_learn.py, train_zo_learn.py: training logic
* compare_gradient.py: compare approximated gradients with true gradients for ablation

# Installation
* Need to install CUDA support for photonic tensor core acceleration, including MODULE=`hadamard_cuda`, `matrix_parametrization`, and `universal_cuda`\
`> cd pyutils/cuda_extension/MODULE`\
`> python3 setup.py install --user`

# Usage
* Pretrain model.\
`> python3 train_pretrain.py config/cifar10/vgg8/pretrain.yml`

* Identity calibration and parallel mapping. Please set your hyperparameters in CONFIG=`config/cifar10/vgg8/pm/pm.yml` and run\
`> python3 train_map.py CONFIG --checkpoint.restore_checkpoint=path/to/your/pretrained/checkpoint`

* Subspace learning with multi-level sampling. Please set your hyperparameters in CONFIG=`config/cifar10/vgg8/ds/learn.yml` and run\
`> python3 train_learn.py CONFIG --checkpoint.restore_chekcpoint=path/to/your/mapped/checkpoint --checkpoint.resume=1`

* All scripts for experiments are in `./script`. For example, to run subspace learning with feedback sampling, column sampling, and data sampling, you can write proper task setting in SCRIPT=`script/vgg8/train_ds_script.py` and run\
`> python3 SCRIPT`

* Comparison experiments with RAD [ICLR 2021] and SWAT-U [NeurIPS 2020]. Run with the SCRIPT=`script/vgg8/train_rad_script.py` and `script/vgg8/train_swat_script.py`,\
`> python3 SCRIPT`

* Comparison with FLOPS [DAC 2020] and MixedTrn [AAAI 2021]. Run with the METHOD=`mixedtrain` or `flops`,\
`> python3 train_zo_learn.py config/mnist/cnn3/METHOD/learn.yml`
