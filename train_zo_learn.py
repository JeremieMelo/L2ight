#!/usr/bin/env python
# coding=UTF-8
import argparse
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
from pyutils.torch_train import (BestKModelSaver, count_parameters,
                                 get_learning_rate, load_model,
                                 set_torch_deterministic)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler


def train(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Scheduler,
        epoch: int,
        criterion: Criterion,
        teacher: nn.Module,
        device: torch.device) -> None:
    model.train()
    step = epoch * len(train_loader)
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        break
    for batch_idx, _ in enumerate(train_loader):
        if(configs.sparse.bp_data_alg == "smd"):
            if np.random.rand() < configs.sparse.bp_data_sparsity:
                continue
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output, loss = optimizer.step(data, target)

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

        step += 1

        if batch_idx % int(configs.run.log_interval) == 0:
            lg.info('Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            mlflow.log_metrics({"train_loss": loss.item()}, step=step)

    scheduler.step()
    accuracy = 100. * correct.float() / len(train_loader.dataset)
    lg.info(
        f"Train Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f})%")
    mlflow.log_metrics({"train_acc": accuracy.item(),
                        "lr": get_learning_rate(optimizer)}, step=epoch)


def validate(
        model: nn.Module,
        validation_loader: DataLoader,
        epoch: int,
        criterion: Criterion,
        loss_vector: Iterable,
        accuracy_vector: Iterable,
        device: torch.device) -> None:
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

    accuracy = 100. * correct.float() / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    lg.info('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))
    mlflow.log_metrics({"val_acc": accuracy.data.item(),
                        "val_loss": val_loss}, step=epoch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if (torch.cuda.is_available() and int(configs.run.use_cuda)):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device('cuda:'+str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        torch.backends.cudnn.benchmark = False

    if(int(configs.run.deterministic) == True):
        set_torch_deterministic()


    model = builder.make_model(device, int(configs.noise.random_state) if int(configs.run.deterministic) else None)

    train_loader, validation_loader = builder.make_dataloader()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)
    criterion = builder.make_criterion().to(device)
    saver = BestKModelSaver(k=int(configs.checkpoint.save_best_model_k))

    lg.info(f'Number of parameters: {count_parameters(model)}')

    model_name = f"{configs.model.name}_wb-{configs.quantize.weight_bit}_ib-{configs.quantize.input_bit}_bpds-{configs.sparse.bp_data_sparsity}_bprank-{configs.sparse.bp_rank}_bpfw-{configs.sparse.bp_forward_weight_sparsity}_bpbw-{configs.sparse.bp_feedback_weight_sparsity}_bpin-{configs.sparse.bp_input_sparsity}_bpsp-{configs.sparse.bp_spatial_sparsity}_bpcol-{configs.sparse.bp_column_sparsity}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    mlflow.set_experiment(configs.run.experiment)
    experiment = mlflow.get_experiment_by_name(configs.run.experiment)

    # run_id_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mlflow.start_run(run_name=model_name)
    mlflow.log_params({
        "exp_name": configs.run.experiment,
        "exp_id": experiment.experiment_id,
        "run_id": mlflow.active_run().info.run_id,
        "inbit": configs.quantize.input_bit,
        "wbit": configs.quantize.weight_bit,
        "init_lr": configs.optimizer.lr,
        "checkpoint": checkpoint,
        "restore_checkpoint": configs.checkpoint.restore_checkpoint,
        "pid": os.getpid()
    })

    lossv, accv = [0], [0]
    epoch = 0
    try:
        lg.info(
            f"Experiment {configs.run.experiment} ({experiment.experiment_id}) starts. Run ID: ({mlflow.active_run().info.run_id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})")
        lg.info(configs)
        lg.info("Model subspace learning...")
        if(int(configs.checkpoint.resume)):
            load_model(model, path=configs.checkpoint.restore_checkpoint)
            model.switch_mode_to("phase")
            model.sync_parameters(src="weight")

            lg.info("Validate loaded ideal model without noise...")
            ### this is important, to call build_weight to construct non-ideal U and V
            validate(
                model,
                validation_loader,
                0,
                criterion,
                lossv,
                accv,
                device)
        # inject non-ideality
        # deterministic phase bias
        if(configs.noise.phase_bias):
            model.assign_random_phase_bias(random_state=int(configs.noise.random_state))
        # deterministic phase shifter gamma noise
        model.set_gamma_noise(float(configs.noise.gamma_noise_std),
                              random_state=int(configs.noise.random_state))
        # deterministic phase shifter crosstalk
        model.set_crosstalk_factor(float(configs.noise.crosstalk_factor))
        # deterministic phase quantization
        model.set_weight_bitwidth(int(configs.quantize.weight_bit))
        # enable/disable noisy identity
        model.set_noisy_identity(int(configs.sl.noisy_identity))
        # set sparsity
        model.set_bp_feedback_sampler(float(configs.sparse.bp_forward_weight_sparsity),
                                      float(configs.sparse.bp_feedback_weight_sparsity),
                                      alg=configs.sparse.bp_feedback_alg,
                                      normalize=configs.sparse.bp_feedback_norm,
                                      random_state=None)
        model.set_bp_input_sampler(float(configs.sparse.bp_input_sparsity),
                                   float(configs.sparse.bp_spatial_sparsity),
                                   float(configs.sparse.bp_column_sparsity),
                                   normalize=configs.sparse.bp_input_norm,
                                   random_state=None,
                                   sparsify_first_conv=int(configs.sparse.bp_input_sparsify_first_conv))
        model.set_bp_rank_sampler(int(configs.sparse.bp_rank), alg=configs.sparse.bp_rank_alg,
                                  sign=int(configs.sparse.bp_rank_sign), random_state=None)

        lg.info("Validate loaded model with noise...")
        validate(
            model,
            validation_loader,
            0,
            criterion,
            lossv,
            accv,
            device)
        model.switch_mode_to("usv")

        # build teacher model
        if(len(configs.checkpoint.restore_checkpoint_pretrained) > 0):
            import copy
            teacher = copy.deepcopy(model)
            load_model(
                teacher, configs.checkpoint.restore_checkpoint_pretrained)
            teacher.switch_mode_to("weight")
            teacher.eval()
        else:
            teacher = None

        model.reset_learning_profiling()
        report = model.get_learning_profiling(flat=True, input_size=(configs.run.batch_size, configs.dataset.in_channel, configs.dataset.img_height, configs.dataset.img_width))
        lg.info(report)
        for epoch in range(1, int(configs.run.n_epochs)+1):
            train(
                model,
                train_loader,
                optimizer,
                scheduler,
                epoch,
                criterion,
                teacher,
                device)
            validate(
                model,
                validation_loader,
                epoch,
                criterion,
                lossv,
                accv,
                device)
            saver.save_model(
                model,
                accv[-1],
                epoch=epoch,
                path=checkpoint,
                save_model=False,
                print_msg=True
            )
            report = model.get_learning_profiling(flat=True, input_size=(configs.run.batch_size, configs.dataset.in_channel, configs.dataset.img_height, configs.dataset.img_width))
            lg.info(report)
            mlflow.log_metrics(report, step=epoch)

        report = model.get_learning_profiling(input_size=(configs.run.batch_size, configs.dataset.in_channel, configs.dataset.img_height, configs.dataset.img_width))
        lg.info(report)
        mlflow.log_dict(report, "profile.yaml")
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
