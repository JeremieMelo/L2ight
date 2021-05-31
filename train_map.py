#!/usr/bin/env python
# coding=UTF-8
import argparse
import os
from typing import Iterable

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchpack.utils.config import configs

from core import builder
from pyutils.general import logger as lg
from pyutils.torch_train import (BestKModelSaver, count_parameters,
                                 get_learning_rate, load_model, set_torch_deterministic)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler



def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    device: torch.device) -> None:
    model.train()
    step = epoch * len(train_loader)
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()

        output = model(data)

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

        classify_loss = criterion(output, target)

        loss = classify_loss

        loss.backward()

        optimizer.step()
        step += 1

        if batch_idx % configs.run.log_interval == 0:
            lg.info('Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4f} Class Loss: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item(), classify_loss.data.item()))
            mlflow.log_metrics({"train_loss": loss.item()}, step=step)

    scheduler.step()
    accuracy = 100. * correct.float() / len(train_loader.dataset)
    lg.info(f"Train Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f})%")
    mlflow.log_metrics({"train_acc": accuracy.data.item(), "lr": get_learning_rate(optimizer)}, step=epoch)


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
    mlflow.log_metrics({"val_acc": accuracy.data.item(), "val_loss": val_loss}, step=epoch)


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

    if(configs.run.deterministic == True):
        set_torch_deterministic()

    model = builder.make_model(device)
    train_loader, validation_loader = builder.make_dataloader()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)
    criterion = builder.make_criterion().to(device)
    saver = BestKModelSaver(k=configs.checkpoint.save_best_model_k)

    lg.info(f'Number of parameters: {count_parameters(model)}')

    model_name = f"{configs.model.name}_wb-{configs.quantize.weight_bit}_ib-{configs.quantize.input_bit}_icalg-{configs.ic.alg}_icadapt-{configs.ic.adaptive}_icbest-{configs.ic.best_record}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    mlflow.set_experiment(configs.run.experiment)
    experiment = mlflow.get_experiment_by_name(configs.run.experiment)
    mlflow.start_run(run_name=model_name)
    mlflow.log_params({
        "exp_name": configs.run.experiment,
        "exp_id": experiment.experiment_id,
        "run_id":mlflow.active_run().info.run_id,
        "inbit": configs.quantize.input_bit,
        "wbit": configs.quantize.weight_bit,
        "init_lr": configs.optimizer.lr,
        "ic_alg": configs.ic.alg,
        "ic_adapt": configs.ic.adaptive,
        "ic_best_record": configs.ic.best_record,
        "checkpoint": checkpoint,
        "restore_checkpoint": configs.checkpoint.restore_checkpoint,
        "pid": os.getpid()
        })
    lg.info(f"Experiment {configs.run.experiment} ({experiment.experiment_id}) starts. Run ID: ({mlflow.active_run().info.run_id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})")


    lossv, accv = [], []
    epoch = 0
    try:
        lg.info(configs)
        # if(0 and int(configs.checkpoint.no_linear)):
        #     lg.info(f"Load pretrained model {configs.checkpoint.restore_checkpoint}...")
        #     pretrained_dict = torch.load(configs.checkpoint.restore_checkpoint)
        #     model_dict = model.state_dict()
        #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        #     lg.info(f"{set(model_dict.keys())-set(pretrained_dict.keys())} are not loaded")
        #     model_dict.update(pretrained_dict)
        #     model.load_state_dict(model_dict)
        # else:
        #     load_model(model, configs.checkpoint.restore_checkpoint)
        load_model(model, configs.checkpoint.restore_checkpoint, ignore_size_mismatch=int(configs.checkpoint.no_linear))
        lg.info("Validate pre-trained model (MODE = weight)...")
        validate(
                model,
                validation_loader,
                -3,
                criterion,
                [],
                [],
                device)
        model.switch_mode_to("phase")
        model.sync_parameters(src="weight")
        lg.info("Validate converted pre-trained model (MODE = phase)...")
        validate(
                model,
                validation_loader,
                -2,
                criterion,
                [],
                [],
                device)
        ### inject non-ideality
        # deterministic phase bias
        if(int(configs.noise.phase_bias)):
            model.assign_random_phase_bias(random_state=int(configs.noise.random_state))
        # deterministic phase shifter gamma noise
        model.set_gamma_noise(float(configs.noise.gamma_noise_std), random_state=int(configs.noise.random_state))
        # deterministic phase shifter crosstalk
        model.set_crosstalk_factor(float(configs.noise.crosstalk_factor))
        # deterministic phase quantization
        model.set_weight_bitwidth(int(configs.quantize.weight_bit))

        lg.info("Validate converted pre-trained model (MODE = phase) with phase bias...")
        validate(
                model,
                validation_loader,
                -1,
                criterion,
                [],
                [],
                device)

        lg.info("Identity calibration...")
        model.identity_calibration(
            alg=configs.ic.alg,
            lr=float(configs.optimizer.lr),
            n_epochs=int(configs.run.cali_n_epochs),
            lr_gamma=float(configs.scheduler.lr_gamma),
            lr_min=float(configs.scheduler.lr_min),
            adaptive=int(configs.ic.adaptive),
            best_record=int(configs.ic.best_record),
            verbose=int(configs.debug.verbose))

        lg.info("Parallel mapping...")

        model.parallel_mapping(
            alg=configs.pm.alg,
            lr=float(configs.optimizer.lr),
            n_epochs=int(configs.run.map_n_epochs),
            lr_gamma=float(configs.scheduler.lr_gamma),
            lr_min=float(configs.scheduler.lr_min),
            adaptive=int(configs.pm.adaptive),
            best_record=int(configs.pm.best_record),
            verbose=int(configs.debug.verbose),
            validate_fn=lambda x:validate(
                model,
                validation_loader,
                -1,
                criterion,
                [],
                x,
                device),
            ideal_I=int(configs.pm.ideal_I))

        lg.info("Validate mapped model...")
        model.sync_parameters(src="phase")
        validate(
                model,
                validation_loader,
                0,
                criterion,
                lossv,
                accv,
                device)
        saver.save_model(
            model,
            accv[-1],
            epoch=int(configs.run.map_n_epochs),
            path=checkpoint,
            save_model=False,
            print_msg=True
        )
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
