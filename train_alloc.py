import argparse
import datetime

import torch.backends.cudnn as cudnn
import json
import yaml
import copy
import re
from pathlib import Path

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler

from timm.utils import NativeScaler
from lib.datasets import build_dataset
from engine import *

import model as models
from timm.models import load_checkpoint
import pickle

import os
from collections import OrderedDict
from lib import utils
import time
from sparse_allocator import Allocator
import optuna
import logging


def get_args_parser():
    parser = argparse.ArgumentParser('AutoFormer training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--model_name', default=None, type=str)

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')
    # AutoFormer config
    parser.add_argument('--input-size', default=224, type=int)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr-power', type=float, default=1.0,
                        help='power of the polynomial lr scheduler')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='./data/imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--data_percentage', default=1.0, type=float, help='Image Net dataset path')
    parser.add_argument('--data-set', default='IMNET', type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='./',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='path to the pre-trained model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')

    parser.add_argument('--no_aug', action='store_true')
    parser.add_argument('--val_interval', default=1, type=int, help='validataion interval')
    parser.add_argument('--inception',action='store_true')
    parser.add_argument('--direct_resize',action='store_true')

    # SNELL params
    parser.add_argument('--freeze_stage', action='store_true')
    parser.add_argument('--scaler', default='naive', type=str,)
    parser.add_argument('--low_rank_dim', default=8, type=int, help='The rank of Adapter or LoRA')
    parser.add_argument('--init_thres', default=0, type=float, help='hyper-parameter, the easiness level for a vector to be structurally tuned.')
    parser.add_argument('--norm_p', default=2, type=float, help='hyper-parameter, the easiness level for a vector to be structurally tuned.')
    parser.add_argument('--no_drop_out', action='store_true')
    parser.add_argument('--no_drop_path', action='store_true')

    # Sparsity allocator
    parser.add_argument('--use_sparse_allocator', action='store_true')
    parser.add_argument('--target_ratio', default=0.9, type=float,)
    parser.add_argument('--init_warmup', default=10, type=int,)
    parser.add_argument('--final_warmup', default=400, type=int,)
    parser.add_argument('--mask_interval', default=20, type=int)
    parser.add_argument('--beta1', default=0.85, type=float,)
    parser.add_argument('--beta2', default=0.85, type=float,)
    parser.add_argument('--metric', default='ipt', type=str,)
    
    parser.add_argument('--freeze_kwd', default='patch_embed', type=str, help='freeze patch embedding helps')
    parser.add_argument('--test', action='store_true', help='using test-split or validation split')
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument('--block', type=str, default='BlockSPTParallel')
    parser.add_argument('--exp_name', default='', type=str)
    parser.add_argument('--tuning_model', type=str, default='SNELL', help="tuning model to use")

    parser.add_argument('--local_rank', default=0, type=int,)

    return parser


def main(args):


    print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)


    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args,)
    dataset_val, _ = build_dataset(is_train=False, args=args,)


    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=512,
        sampler=sampler_val, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    print(f"{args.data_set} dataset, train: {len(dataset_train)}, evaluation: {len(dataset_val)}")

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    print('mixup_active',mixup_active)
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    


    fully_fine_tuned_keys = []
    fully_fine_tuned_keys.extend(['head.weight', 'head.bias', 'cls_token'])

    model = models.__dict__[args.model_name](img_size=args.input_size,
                                                drop_rate=args.drop,
                                                drop_path_rate=args.drop_path,
                                                freeze_backbone=args.freeze_stage,
                                                low_rank_dim=args.low_rank_dim,
                                                block=args.block,
                                                num_classes=args.nb_classes,
                                                tuning_model=args.tuning_model,
                                                no_drop_out=args.no_drop_out,
                                                no_drop_path=args.no_drop_path, 
                                                init_thres=args.init_thres,
                                                norm_p=args.norm_p
                                                )

    train_engine = train_one_epoch
    test_engine = evaluate
    total_param = 0
    for name, param in model.named_parameters():
        total_param += param.numel()
    

    if args.resume:
        # Hard-coded pre-trained model name
        if '.pth' in args.resume:

            if args.resume.endswith('mae_pretrain_vit_base.pth'):
                state_dict = torch.load(args.resume, map_location='cpu')['model']
                new_dict = OrderedDict()
                for name in state_dict.keys():
                    if 'attn.qkv.' in name:
                        new_dict[name.replace('qkv', 'q')] = state_dict[name][:state_dict[name].shape[0] // 3]
                        new_dict[name.replace('qkv', 'k')] = state_dict[name][state_dict[name].shape[0] // 3:-state_dict[name].shape[0] // 3]
                        new_dict[name.replace('qkv', 'v')] = state_dict[name][-state_dict[name].shape[0] // 3:]
                    else:
                        new_dict[name] = state_dict[name]

                msg = model.load_state_dict(new_dict, strict=False)
                print('Resuming from MAE model: ', msg)

            elif args.resume.endswith('linear-vit-b-300ep.pth.tar'):
                state_dict = torch.load(args.resume, map_location='cpu')['state_dict']
                new_dict = OrderedDict()
                for name in state_dict.keys():
                    if 'attn.qkv.' in name:
                        new_dict[name.replace('qkv', 'q').split('module.')[1]] = state_dict[name][:state_dict[name].shape[0] // 3]
                        new_dict[name.replace('qkv', 'k').split('module.')[1]] = state_dict[name][state_dict[name].shape[0] // 3:-state_dict[name].shape[0] // 3]
                        new_dict[name.replace('qkv', 'v').split('module.')[1]] = state_dict[name][-state_dict[name].shape[0] // 3:]
                    elif 'head.' in name:
                        continue
                    else:
                        new_dict[name.split('module.')[1]] = state_dict[name]

                msg = model.load_state_dict(new_dict, strict=False)
                print('Resuming from MoCo model: ', msg)

            elif args.resume.endswith('swin_base_patch4_window7_224_22k.pth'):

                state_dict = torch.load(args.resume, map_location='cpu')['model']
                new_dict = OrderedDict()
                for name in state_dict.keys():
                    #if 'attn.qkv.' in name:
                    #    new_dict[name.replace('qkv', 'q')] = state_dict[name][:state_dict[name].shape[0] // 3]
                    #    new_dict[name.replace('qkv', 'k')] = state_dict[name][state_dict[name].shape[0] // 3:-state_dict[name].shape[0] // 3]
                    #    new_dict[name.replace('qkv', 'v')] = state_dict[name][-state_dict[name].shape[0] // 3:]
                    #elif 'head.' in name:
                    if 'head.' in name:
                        continue
                    else:
                        new_dict[name] = state_dict[name]

                if args.nb_classes != model.head.weight.shape[0]:
                    model.reset_classifier(args.nb_classes)

                msg = model.load_state_dict(new_dict, strict=False)
                #msg = model.load_state_dict(state_dict, strict=False)
                print('Resuming from Swin model: ', msg)

            elif args.resume.endswith('convnext_base_22k_224.pth'):

                state_dict = torch.load(args.resume, map_location='cpu')['model']
                new_dict = OrderedDict()
                for k, v in state_dict.items():
                    if 'head.' in k:
                        continue
                    k = k.replace('downsample_layers.0.0.', 'stem.proj.')
                    k = k.replace('downsample_layers.0.1.', 'stem.norm.')


                    k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)

                    k = re.sub(r'downsample_layers.([0-9]+).([0]+)', r'stages.\1.downsample.norm', k)
                    k = re.sub(r'downsample_layers.([0-9]+).([1]+)', r'stages.\1.downsample.proj', k)


                    k = k.replace('dwconv', 'conv_dw')
                    k = k.replace('pwconv', 'mlp.fc')
                    k = k.replace('head.', 'head.fc.')
                    if k.startswith('norm.'):
                        k = k.replace('norm', 'head.norm')
                    if v.ndim == 2 and 'head' not in k:
                        model_shape = model.state_dict()[k].shape
                        v = v.reshape(model_shape)
                    new_dict[k] = v

                #if args.nb_classes != model.head.weight.shape[0]:
                model.reset_classifier(args.nb_classes)

                msg = model.load_state_dict(new_dict, strict=False)
                #msg = model.load_state_dict(state_dict, strict=False)
                print('Resuming from convnext model: ', msg)
            else:
                raise NotImplementedError

        else:
            load_checkpoint(model, args.resume)
            print(f'load from {args.resume}')
            if args.nb_classes != model.head.weight.shape[0]:
                model.reset_classifier(args.nb_classes)

    model.to(device)
    
    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = utils.build_optimizer(args, model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # save config for later experiments
    with open(output_dir / "config.yaml", 'w') as f:
        f.write(args_text)

    if args.eval:
        test_stats = test_engine(data_loader_val, model, device, amp=args.amp)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print("Start training")

    start_time = time.time()
    max_accuracy = 0.0

    allocator = None
    if args.use_sparse_allocator:
        print('utilizing allocator')
        print(f'total steps: {args.epochs}')
        allocator = Allocator(model, args.target_ratio, args.init_warmup, args.final_warmup, args.mask_interval, args.beta1, args.beta2, args.epochs)

    best_model = None
    for epoch in range(args.start_epoch, args.epochs):

        
        if allocator is not None:
            allocator.update(model, epoch)

        train_stats = train_engine(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            amp=args.amp, scaler=args.scaler, allocator=allocator, metric=args.metric,
        )
        lr_scheduler.step(epoch)

        if epoch % args.val_interval == 0 or epoch >= args.epochs-10:  # Evaluate more in the last a few epochs
            test_stats = test_engine(data_loader_val, model, device, amp=args.amp)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats['acc1']:
                best_model = copy.deepcopy(model)
                max_accuracy = test_stats["acc1"]
            print(
                f"[{args.exp_name}] Max accuracy on the {args.data_set} dataset {len(dataset_val)} with ({args.opt}, {args.lr}, {args.weight_decay}), {max_accuracy:.2f}%")

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,}

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                if allocator is not None:
                    sparsity_pattern = allocator.get_sparsity_pattern()
                    score_pattern = allocator.get_score_pattern()
                    with (output_dir / "sparsity_pattern.txt").open("a") as f:
                        f.write(json.dumps(sparsity_pattern) + "\n")
                    with (output_dir / "score_pattern.txt").open("a") as f:
                        f.write(json.dumps(score_pattern) + "\n")
            if args.save_best:
                torch.save(best_model.state_dict(), f'{output_dir}/{max_accuracy}_model_best.pth')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if args.use_sparse_allocator:
        alloc_volume_dict = allocator.get_sparsity_pattern()
        with open(os.path.join(args.output_dir, 'alloc_volume_dict.json'), 'w') as f:
            json.dump(alloc_volume_dict, f)
        score_volume_dict = allocator.get_score_pattern()
        with open(os.path.join(args.output_dir, 'score_volume_dict.json'), 'w') as f:
            json.dump(score_volume_dict, f)
    return max_accuracy
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SNELL training and evaluation scripts', parents=[get_args_parser()])
    args = parser.parse_args()

    model_e_name = args.model_name.split('_')[0]

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
    