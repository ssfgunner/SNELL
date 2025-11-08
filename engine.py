import math
import sys
from typing import Iterable, Optional
import torch
import torch.nn as nn
import torch.distributed as dist

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
import random
import numpy as np
import os
from tqdm import tqdm

from collections import defaultdict
# All Structured positions
vit_operation_dict = {'q': 0, 'k': 1, 'v': 2, 'proj': 3, 'fc1': 4, 'fc2': 5}


TARGET_LAYER_LIST = [
    'blocks.0',
    'blocks.1',
    'blocks.2',
    'blocks.3',
    'blocks.4',
    'blocks.5',
    'blocks.6',
    'blocks.7',
    'blocks.8',
    'blocks.9',
    'blocks.10',
    'blocks.11',
]

def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, 
                    mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, scaler=None, allocator=None, metric='ipt', model_ema=None, model_without_ddp=None):

    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    current_step = epoch * len(data_loader)
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        #gpu_tracker.track()
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if amp:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)
        else:
            outputs = model(samples)
            loss = criterion(outputs, targets)
        

        loss_value = loss.item()


        if not math.isfinite(loss_value):
            if is_main_process():
                print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)

        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        elif scaler != 'naive':
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    model=model, create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()
        if model_ema is not None:
            model_ema.update(model)
        
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(name)
        
        if allocator is not None:
            
            #allocator.update(model, current_step)
            if model_without_ddp is None:
                model_without_ddp = model
            allocator.update_ipt(model_without_ddp, metric=metric)
            

        current_step += 1
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if is_main_process():
        print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_structured_param_num(structured_type=None, in_dim=768, out_dim=768, low_rank_dim=8):
    if structured_type =='lora':
        return in_dim * low_rank_dim + low_rank_dim * out_dim
    elif structured_type =='adapter':
        return out_dim * low_rank_dim + low_rank_dim * out_dim + low_rank_dim + out_dim
    else:
        raise NotImplementedError


@torch.no_grad()
def evaluate(data_loader, model, device, amp=True):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        try:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        except RuntimeError:
            # class_num <= 5
            acc1 = accuracy(output, target, topk=(1,))
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1[0].item(), n=batch_size)
            metric_logger.meters['acc5'].update(0., n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if is_main_process():
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
