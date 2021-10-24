"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:08:51
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-24 16:08:51
"""
#!/usr/bin/env python
# coding=UTF-8
import argparse
import os
from typing import Dict, Iterable, List

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.types import Device
from pyutils.config import configs

from core import builder
from core.models.layers.custom_conv2d import MZIBlockConv2d
from core.models.layers.custom_linear import MZIBlockLinear
from pyutils.general import logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler


def set_sparsity(model: nn.Module) -> None:
    # set sparsity
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
    )
    model.set_bp_rank_sampler(
        int(configs.sparse.bp_rank),
        alg=configs.sparse.bp_rank_alg,
        sign=int(configs.sparse.bp_rank_sign),
        random_state=None,
    )


def reset_sparsity(model: nn.Module) -> None:
    # reset sparsity
    model.set_bp_feedback_sampler(
        0, 0, alg=configs.sparse.bp_feedback_alg, normalize=configs.sparse.bp_feedback_norm, random_state=None
    )
    model.set_bp_input_sampler(0, 0, 0, normalize=configs.sparse.bp_input_norm, random_state=None)
    model.set_bp_rank_sampler(
        configs.model.block_list[0], alg=configs.sparse.bp_rank_alg, sign=0, random_state=None
    )
    model.set_noisy_identity(False)


def extract_gradients(model: nn.Module):
    grads = {}
    for name, m in model.named_modules():
        if isinstance(m, (MZIBlockConv2d, MZIBlockLinear)):
            grads[name] = m.S.grad.data.clone().flatten(0)  # [p, q, k] -> [pqk]
    total_grad = torch.cat(list(grads.values()), dim=0)
    return grads, total_grad


def cosine_distance(grads_1: Dict[str, Tensor], grads_2: Dict[str, Tensor]) -> Dict[str, float]:
    dist = np.zeros([len(grads_1)])
    for idx, name in enumerate(grads_1):
        dist[idx] = (1 - F.cosine_similarity(grads_1[name], grads_2[name], dim=0)).cpu().item()
    return dist


def l2_distance(
    grads_1: Dict[str, Tensor], total_grad_1: Tensor, grads_2: Dict[str, Tensor], total_grad_2: Tensor
) -> Dict[str, float]:
    dist = np.zeros([len(grads_1)])
    for idx, name in enumerate(grads_1):
        dist[idx] = (
            (grads_1[name].sub(grads_2[name]).square_().sum().div(grads_1[name].norm(2).square()))
            .cpu()
            .item()
        )
    total_dist = (
        (total_grad_1.sub(total_grad_2).square_().sum().div(total_grad_1.norm(2).square())).cpu().item()
    )

    return dist, total_dist


def angular_similarity(
    grads_1: Dict[str, Tensor], total_grad_1: Tensor, grads_2: Dict[str, Tensor], total_grad_2: Tensor
) -> Dict[str, float]:
    dist = np.zeros([len(grads_1)])
    for idx, name in enumerate(grads_1):
        dist[idx] = (
            (1 - F.cosine_similarity(grads_1[name], grads_2[name], dim=0).acos_().div_(np.pi)).cpu().item()
        )
    total_dist = (1 - F.cosine_similarity(total_grad_1, total_grad_2, dim=0).acos_().div_(np.pi)).cpu().item()
    return dist, total_dist


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    device: Device,
) -> None:
    model.train()
    step = epoch * len(train_loader)
    correct = 0
    accum_arg_dist = []
    accum_l2_dist = []
    accum_total_arg_dist = []
    accum_total_l2_dist = []

    for batch_idx, (data, target) in enumerate(train_loader):
        if np.random.rand() < configs.sparse.bp_data_sparsity:
            continue
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # original gradient
        model.zero_grad()
        reset_sparsity(model)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        grads_ori, total_grad_ori = extract_gradients(model)

        # sampled gradient
        model.zero_grad()
        set_sparsity(model)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        grads_sampled, total_grad_sampled = extract_gradients(model)

        # calculate distance
        arg_dist, total_arg_dist = angular_similarity(
            grads_ori, total_grad_ori, grads_sampled, total_grad_sampled
        )
        l2_dist, total_l2_dist = l2_distance(grads_ori, total_grad_ori, grads_sampled, total_grad_sampled)

        accum_arg_dist.append(arg_dist)
        accum_l2_dist.append(l2_dist)
        accum_total_arg_dist.append(total_arg_dist)
        accum_total_l2_dist.append(total_l2_dist)

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

        classify_loss = loss

        optimizer.step()
        step += 1

        if batch_idx % int(configs.run.log_interval) == 0:
            lg.info(
                "Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4f} Class Loss: {:.4f} angular: {:.3f} L2: {:.3f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                    classify_loss.data.item(),
                    total_arg_dist,
                    total_l2_dist,
                )
            )
            mlflow.log_metrics({"train_loss": loss.item()}, step=step)

    accum_arg_dist = np.stack(accum_arg_dist, 0)
    accum_l2_dist = np.stack(accum_l2_dist, 0)
    accum_total_arg_dist = np.stack(accum_total_arg_dist, 0)
    accum_total_l2_dist = np.stack(accum_total_l2_dist, 0)
    layer_avg_arg_dist, layer_std_arg_dist = np.mean(accum_arg_dist, 0), np.std(accum_arg_dist, 0)
    layer_avg_l2_dist, layer_std_l2_dist = np.mean(accum_l2_dist, 0), np.std(accum_l2_dist, 0)
    total_avg_arg_dist, total_std_arg_dist = np.mean(accum_total_arg_dist), np.std(accum_total_arg_dist)
    total_avg_l2_dist, total_std_l2_dist = np.mean(accum_total_l2_dist), np.std(accum_total_l2_dist)
    mlflow.log_metrics(
        {
            "total_avg_arg": total_avg_arg_dist,
            "total_std_arg": total_std_arg_dist,
            "total_avg_l2": total_avg_l2_dist,
            "total_std_l2": total_std_l2_dist,
        },
        step=epoch,
    )

    for i in range(len(layer_avg_arg_dist)):
        mlflow.log_metrics(
            {
                f"l{i}_avg_arg": layer_avg_arg_dist[i],
                f"l{i}_std_arg": layer_std_arg_dist[i],
                f"l{i}_avg_l2": layer_avg_l2_dist[i],
                f"l{i}_std_l2": layer_std_l2_dist[i],
            },
            step=epoch,
        )

    scheduler.step()
    accuracy = 100.0 * correct.float() / len(train_loader.dataset)
    lg.info(f"Train Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f})%")
    mlflow.log_metrics({"train_acc": accuracy.data.item(), "lr": get_learning_rate(optimizer)}, step=epoch)


def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: Device,
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

    if torch.cuda.is_available() and configs.run.use_cuda:
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if configs.run.deterministic == True:
        set_torch_deterministic()

    model = builder.make_model(device)
    train_loader, validation_loader = builder.make_dataloader()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)
    criterion = builder.make_criterion().to(device)
    saver = BestKModelSaver(k=configs.checkpoint.save_best_model_k)

    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}_wb-{configs.quantize.weight_bit}_ib-{configs.quantize.input_bit}_bpds-{configs.sparse.bp_data_sparsity}_bprank-{configs.sparse.bp_rank}_bpfw-{configs.sparse.bp_forward_weight_sparsity}_bpbw-{configs.sparse.bp_feedback_weight_sparsity}_bpin-{configs.sparse.bp_input_sparsity}_bpsp-{configs.sparse.bp_spatial_sparsity}_bpcol-{configs.sparse.bp_column_sparsity}"
    checkpoint = (
        f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"
    )

    lg.info(f"Current checkpoint: {checkpoint}")

    mlflow.set_experiment(configs.run.experiment)
    experiment = mlflow.get_experiment_by_name(configs.run.experiment)
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
            f"Experiment {configs.run.experiment} ({experiment.experiment_id}) starts. Run ID: ({mlflow.active_run().info.run_id}). PID: ({os.getpid()})"
        )
        lg.info(configs)
        lg.info("Model subspace learning...")
        if configs.checkpoint.resume:
            load_model(model, path=configs.checkpoint.restore_checkpoint)
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
        )
        model.set_bp_rank_sampler(
            int(configs.sparse.bp_rank),
            alg=configs.sparse.bp_rank_alg,
            sign=int(configs.sparse.bp_rank_sign),
            random_state=None,
        )

        lg.info("Validate mapped model...")
        validate(model, validation_loader, 0, criterion, lossv, accv, device)

        for epoch in range(1, configs.run.n_epochs + 1):
            train(model, train_loader, optimizer, scheduler, epoch, criterion, device)
            validate(model, validation_loader, epoch, criterion, lossv, accv, device)
            saver.save_model(model, accv[-1], epoch=epoch, path=checkpoint, save_model=False, print_msg=True)

    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
