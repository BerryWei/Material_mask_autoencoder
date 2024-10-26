# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from torcheval.metrics import R2Score

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    r2_metric = R2Score().to(device)

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if samples.shape[1] == 3:
            samples = samples.mean(dim=1, keepdim=True)  # 將 RGB 轉換為灰階

        with torch.amp.autocast('cuda'):
            outputs = model(samples)
            outputs = outputs.squeeze(dim=-1)  # 將 [batch_size, 1] 轉為 [batch_size]
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        
        r2_metric.update(outputs, targets)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    r2_value = r2_metric.compute()
    metric_logger.update(r2=r2_value)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    r2_metric.reset()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.MSELoss()
    mae_criterion = torch.nn.L1Loss()
    r2_metric = R2Score().to(device)

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)  # 將 RGB 轉換為灰階

        # compute output
        with torch.amp.autocast('cuda'):
            output = model(images)
            output = output.squeeze(dim=-1)  # 將 [batch_size, 1] 轉為 [batch_size]

            loss = criterion(output, target)

        # 計算 MAE
        mae_loss = mae_criterion(output, target)
        r2_metric.update(output, target)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['mae'].update(mae_loss.item(), n=batch_size)
        metric_logger.meters['mse'].update(loss.item(), n=batch_size)
    # gather the stats from all processes

    r2_value = r2_metric.compute()
    metric_logger.synchronize_between_processes()
    print('* MSE {mse.global_avg:.3f} MAE {mae.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(mse=metric_logger.mse, mae=metric_logger.mae, losses=metric_logger.loss))

    r2_metric.reset()

    result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    result['r2'] = r2_value.item()
    return result

    