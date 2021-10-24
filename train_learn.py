"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:08:04
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-24 16:08:05
"""
#!/usr/bin/env python
# coding=UTF-8
import argparse
from core.models.layers.custom_conv2d import MZIBlockConv2d
import datetime
import os
from typing import Iterable

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyutils.config import configs

from core import builder
from pyutils.general import logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    teacher: nn.Module,
    device: torch.device,
) -> None:
    model.train()
    step = epoch * len(train_loader)
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if configs.sparse.bp_data_alg == "smd":
            if np.random.rand() < float(configs.sparse.bp_data_sparsity):
                continue
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()

        if (
            teacher is not None
            and configs.sparse.bp_data_alg == "is"
            and float(configs.sparse.bp_data_sparsity) > 0
        ):
            with torch.no_grad():
                teacher_score = F.softmax(teacher(data.data), dim=-1)

                teacher_entropy = teacher_score.mul(-(teacher_score + 1e-12).log()).sum(dim=-1)
            mask = teacher_entropy > torch.quantile(
                teacher_entropy, float(configs.sparse.bp_data_sparsity)
            )
            if not mask.any():
                continue
            data = data[mask, :]
            target = target[mask]
            output = model(data)
            classify_loss = criterion(output, target)
        else:
            output = model(data)
            classify_loss = criterion(output, target)

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

        loss = classify_loss

        loss.backward()

        optimizer.step()
        step += 1

        if batch_idx % int(configs.run.log_interval) == 0:
            lg.info(
                "Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4f} Class Loss: {:.4f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                    classify_loss.data.item(),
                )
            )
            mlflow.log_metrics({"train_loss": loss.item()}, step=step)

    scheduler.step()
    accuracy = 100.0 * correct.float() / len(train_loader.dataset)
    lg.info(f"Train Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f})%")
    mlflow.log_metrics({"train_acc": accuracy.item(), "lr": get_learning_rate(optimizer)}, step=epoch)


def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
) -> None:
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in validation_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)

            val_loss += criterion(output, target).data.item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100.0 * correct.float() / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    lg.info(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            val_loss, correct, len(validation_loader.dataset), accuracy
        )
    )
    mlflow.log_metrics({"val_acc": accuracy.data.item(), "val_loss": val_loss}, step=epoch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if int(configs.run.deterministic) == True:
        set_torch_deterministic()

    model = builder.make_model(
        device, int(configs.run.random_state) if int(configs.run.deterministic) else None
    )

    train_loader, validation_loader = builder.make_dataloader()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)
    criterion = builder.make_criterion().to(device)
    saver = BestKModelSaver(k=int(configs.checkpoint.save_best_model_k))

    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}_wb-{configs.quantize.weight_bit}_ib-{configs.quantize.input_bit}_bpds-{configs.sparse.bp_data_sparsity}_bprank-{configs.sparse.bp_rank}_bpfw-{configs.sparse.bp_forward_weight_sparsity}_bpbw-{configs.sparse.bp_feedback_weight_sparsity}_bpin-{configs.sparse.bp_input_sparsity}_bpsp-{configs.sparse.bp_spatial_sparsity}_bpcol-{configs.sparse.bp_column_sparsity}"
    checkpoint = (
        f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"
    )

    lg.info(f"Current checkpoint: {checkpoint}")

    mlflow.set_experiment(configs.run.experiment)
    experiment = mlflow.get_experiment_by_name(configs.run.experiment)

    # run_id_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mlflow.start_run(run_name=model_name)
    mlflow.log_params(
        {
            "exp_name": configs.run.experiment,
            "exp_id": experiment.experiment_id,
            "run_id": mlflow.active_run().info.run_id,
            "inbit": configs.quantize.input_bit,
            "wbit": configs.quantize.weight_bit,
            "init_lr": configs.optimizer.lr,
            "checkpoint": checkpoint,
            "restore_checkpoint": configs.checkpoint.restore_checkpoint,
            "pid": os.getpid(),
        }
    )

    lossv, accv = [0], [0]
    epoch = 0
    try:
        lg.info(
            f"Experiment {configs.run.experiment} ({experiment.experiment_id}) starts. Run ID: ({mlflow.active_run().info.run_id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})"
        )
        lg.info(configs)
        lg.info("Model subspace learning...")
        if int(configs.checkpoint.resume):
            load_model(
                model,
                configs.checkpoint.restore_checkpoint,
                ignore_size_mismatch=int(configs.checkpoint.no_linear),
            )
        assert model.mode == "usv", lg.error("Only support subspace learning in usv mode.")
        # inject non-ideality
        # deterministic phase bias
        if configs.noise.phase_bias:
            model.assign_random_phase_bias(random_state=int(configs.noise.random_state))
        # deterministic phase shifter gamma noise
        model.set_gamma_noise(
            float(configs.noise.gamma_noise_std), random_state=int(configs.noise.random_state)
        )
        # deterministic phase shifter crosstalk
        model.set_crosstalk_factor(float(configs.noise.crosstalk_factor))
        # deterministic phase quantization
        model.set_weight_bitwidth(int(configs.quantize.weight_bit))
        # enable/disable noisy identity
        model.set_noisy_identity(int(configs.sl.noisy_identity))
        # set sparsity
        model.set_bp_feedback_sampler(
            float(configs.sparse.bp_forward_weight_sparsity),
            float(configs.sparse.bp_feedback_weight_sparsity),
            alg=configs.sparse.bp_feedback_alg,
            normalize=configs.sparse.bp_feedback_norm,
            random_state=None,
        )
        model.set_bp_input_sampler(
            float(configs.sparse.bp_input_sparsity),
            float(configs.sparse.bp_spatial_sparsity),
            float(configs.sparse.bp_column_sparsity),
            normalize=configs.sparse.bp_input_norm,
            random_state=None,
            sparsify_first_conv=int(configs.sparse.bp_input_sparsify_first_conv),
        )
        model.set_bp_rank_sampler(
            int(configs.sparse.bp_rank),
            alg=configs.sparse.bp_rank_alg,
            sign=int(configs.sparse.bp_rank_sign),
            random_state=None,
        )

        if getattr(configs.checkpoint, "imagenet_pretrain", False):
            import torchvision

            model2 = torchvision.models.resnet18(pretrained=True).to(device)
            conv_list1 = [i for i in model.modules() if isinstance(i, MZIBlockConv2d)]
            conv_list2 = [i for i in model2.modules() if isinstance(i, nn.Conv2d)]
            for m1, m2 in zip(conv_list1, conv_list2):
                if m2.weight.size() == (m1.out_channel, m1.in_channel, m1.kernel_size, m1.kernel_size):
                    p, q, k, _ = m1.weight.size()
                    weight = m1.weight.data.permute(0, 2, 1, 3).contiguous().view(p * k, q * k)
                    weight[: m1.out_channel, : m1.in_channel * m1.kernel_size ** 2] = m2.weight.data.flatten(
                        1
                    )
                    m1.weight.data.copy_(weight.view(p, k, q, k).permute(0, 2, 1, 3))
                else:
                    print(m1.weight.size(), m2.weight.size())
                    p, q, k, _ = m1.weight.size()
                    kernel_size1 = m1.kernel_size
                    kernel_size2 = m2.kernel_size[0]
                    left, right = (kernel_size2 - kernel_size1) // 2, -(kernel_size2 - kernel_size1) // 2
                    weight = m1.weight.data.permute(0, 2, 1, 3).contiguous().view(p * k, q * k)
                    weight[: m1.out_channel, : m1.in_channel * m1.kernel_size ** 2] = (
                        m2.weight.data[:, :, left:right, left:right].flatten(1) * kernel_size2 / kernel_size1
                    )
                    m1.weight.data.copy_(weight.view(p, k, q, k).permute(0, 2, 1, 3))

            bn_list1 = [i for i in model.modules() if isinstance(i, nn.BatchNorm2d)]
            bn_list2 = [i for i in model2.modules() if isinstance(i, nn.BatchNorm2d)]
            for m1, m2 in zip(bn_list1, bn_list2):
                m1.weight.data.copy_(m2.weight)
                m1.bias.data.copy_(m2.bias)

            del model2
            model.sync_parameters(src="weight")
            torch.cuda.empty_cache()
            print("Initialize from Imagenet pre-trained ResNet-18")

        lg.info("Validate mapped model...")
        validate(model, validation_loader, 0, criterion, lossv, accv, device)

        # build teacher model
        if len(configs.checkpoint.restore_checkpoint_pretrained) > 0:
            import copy

            teacher = copy.deepcopy(model)
            load_model(teacher, configs.checkpoint.restore_checkpoint_pretrained)
            teacher.switch_mode_to("weight")
            teacher.eval()
        else:
            teacher = None

        model.reset_learning_profiling()
        report = model.get_learning_profiling(
            flat=True,
            input_size=(
                configs.run.batch_size,
                configs.dataset.in_channel,
                configs.dataset.img_height,
                configs.dataset.img_width,
            ),
        )
        lg.info(report)
        set_torch_deterministic(int(configs.sparse.random_state))
        for epoch in range(1, int(configs.run.n_epochs) + 1):
            train(model, train_loader, optimizer, scheduler, epoch, criterion, teacher, device)
            validate(model, validation_loader, epoch, criterion, lossv, accv, device)
            saver.save_model(model, accv[-1], epoch=epoch, path=checkpoint, save_model=False, print_msg=True)
            report = model.get_learning_profiling(
                flat=True,
                input_size=(
                    configs.run.batch_size,
                    configs.dataset.in_channel,
                    configs.dataset.img_height,
                    configs.dataset.img_width,
                ),
            )
            lg.info(report)
            mlflow.log_metrics(report, step=epoch)

        report = model.get_learning_profiling(
            input_size=(
                configs.run.batch_size,
                configs.dataset.in_channel,
                configs.dataset.img_height,
                configs.dataset.img_width,
            )
        )
        lg.info(report)
        mlflow.log_dict(report, "profile.yaml")
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
