import math
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.fx_features import register_notrace_module
from timm.models.helpers import named_apply, build_model_with_cfg
from timm.models.layers import trunc_normal_, ClassifierHead, SelectAdaptivePool2d, DropPath, ConvMlp, to_2tuple

from timm.models.registry import register_model
import torch.utils.checkpoint as cp


__all__ = ['ConvNeXt']  # model_registry will add each entrypoint fn to this


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = dict(
    convnext_tiny=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth"),
    convnext_small=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth"),
    convnext_base=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth"),
    convnext_large=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth"),

    convnext_tiny_hnf=_cfg(url=''),

    convnext_base_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth'),
    convnext_large_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth'),
    convnext_xlarge_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224_ema.pth'),

    convnext_base_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    convnext_large_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    convnext_xlarge_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),

    convnext_base_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth", num_classes=21841),
    convnext_large_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth", num_classes=21841),
    convnext_xlarge_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth", num_classes=21841),
)



def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    # if torch.jit.is_tracing():
    #     return True
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)



@register_notrace_module
class LayerNorm2d(nn.LayerNorm):
    r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x


class KLoRA(nn.Module):
    def __init__(self, in_features, out_features, num_Hs=2, q_dim=16, norm_p=2, SCALE_FACTOR_fc=0.1, enable_bias=False):
        super(KLoRA, self).__init__()
        self.norm_p = norm_p
        self.num_Hs = num_Hs
        self.SCALE_FACTOR_fc = SCALE_FACTOR_fc
        self.In_Qs_1 = torch.nn.Parameter(torch.rand(num_Hs, in_features, q_dim//2))
        self.Out_Qs_1 = torch.nn.Parameter(torch.rand(num_Hs, out_features, q_dim//2))
        self.shared_coeff_fc = torch.nn.Parameter(SCALE_FACTOR_fc*torch.tensor([(-1)**h_id for h_id in range(self.num_Hs)]).unsqueeze(1).unsqueeze(2), requires_grad=False)
        self.bias = None
        self.enable_bias = enable_bias
        if enable_bias:
            self.bias = torch.nn.Parameter(torch.rand(out_features))
            nn.init.constant_(self.bias, 0)
        nn.init.kaiming_uniform_(self.In_Qs_1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Out_Qs_1, a=math.sqrt(5))


    def pathIntegrals(self):
        dist_io = torch.cdist(self.In_Qs_1, self.Out_Qs_1, p=self.norm_p)
        return torch.sum(self.shared_coeff_fc*dist_io, dim=0)

    def _forward(self, x):
        if self.enable_bias:
            return F.linear(x, self.pathIntegrals().T, self.bias)
        else:
            return F.linear(x, self.pathIntegrals().T)

    def forward(self, x):
        return cp.checkpoint(self._forward, x)
        #return self._forward(x)

class SNELL(nn.Module):
    def __init__(self, in_features, out_features, num_Hs=2, init_thres=0, q_dim=16, norm_p=2, SCALE_FACTOR_fc=0.1, enable_bias=False):
        super(SNELL, self).__init__()
        self.norm_p = norm_p
        self.num_Hs = num_Hs
        
        self.In_Qs = torch.nn.Parameter(torch.rand(num_Hs, in_features, q_dim//2))
        self.Out_Qs = torch.nn.Parameter(torch.rand(num_Hs, out_features, q_dim//2))
        self.shared_coeff_fc = torch.nn.Parameter(SCALE_FACTOR_fc*torch.tensor([(-1)**h_id for h_id in range(self.num_Hs)]).unsqueeze(1).unsqueeze(2), requires_grad=False)
        self.bias = None
        self.enable_bias = enable_bias
        if enable_bias:
            self.bias = torch.nn.Parameter(torch.rand(out_features))
            nn.init.constant_(self.bias, 0)

        nn.init.kaiming_uniform_(self.In_Qs, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Out_Qs, a=math.sqrt(5))

        self.init_thres = init_thres
        
    def sparsify(self, weight):
        weight_abs = weight.detach().abs()
        thres = torch.quantile(weight_abs.flatten(), self.init_thres)
        n_sub = F.threshold(weight_abs, float(thres), 0)
        weight = weight * n_sub
        return weight
    

    def pathIntegrals(self):
        dist_io = torch.cdist(self.In_Qs, self.Out_Qs, p=self.norm_p)
        return torch.sum(self.shared_coeff_fc*dist_io, dim=0)
    
    
    def _forward(self, x):
        W_fc = self.pathIntegrals()
        Sparse_W_fc = self.sparsify(W_fc)
        if self.enable_bias:
            return F.linear(x, Sparse_W_fc.T, self.bias)
        else:
            return F.linear(x, Sparse_W_fc.T)
        
    def forward(self, x):
        return cp.checkpoint(self._forward, x)
        #return self._forward(x)


class TuningModule(nn.Module):
    def __init__(self, in_dim, out_dim, low_rank_dim,
                 tuning_model='lora', bias=False, init_thres=0, norm_p=2, **kwargs):

        super().__init__()

        self.tuning_model = tuning_model
        self.low_rank_dim = low_rank_dim

        if tuning_model == 'lora':
            self.learnable = nn.Sequential(
                nn.Linear(in_dim, low_rank_dim, bias=False),
                nn.Linear(low_rank_dim, out_dim, bias=False)
            )

            nn.init.kaiming_uniform_(self.learnable[0].weight, a=math.sqrt(5))
            nn.init.zeros_(self.learnable[1].weight)

        elif tuning_model == 'adapter':
            self.learnable = nn.Sequential(
                nn.Linear(in_dim, low_rank_dim, bias=bias),
                nn.GELU(),
                nn.Linear(low_rank_dim, out_dim, bias=bias)
            )

            nn.init.kaiming_uniform_(self.learnable[0].weight, a=math.sqrt(5))
            nn.init.zeros_(self.learnable[2].weight)
            if bias:
                nn.init.zeros_(self.learnable[2].bias)

        elif tuning_model == 'lora_bias':
            self.learnable = nn.Sequential(
                nn.Linear(in_dim, low_rank_dim, bias=False),
                nn.Linear(low_rank_dim, out_dim, bias=True)
            )

            nn.init.kaiming_uniform_(self.learnable[0].weight, a=math.sqrt(5))
            nn.init.zeros_(self.learnable[1].weight)
            nn.init.zeros_(self.learnable[1].bias)

        elif tuning_model == 'klora':
            self.learnable = KLoRA(in_dim, out_dim, q_dim=low_rank_dim, norm_p=norm_p, enable_bias=False)
        elif tuning_model == 'snell':
            if init_thres == 0:
                self.learnable = KLoRA(in_dim, out_dim, q_dim=low_rank_dim, norm_p=norm_p, enable_bias=False)
            else:
                self.learnable = SNELL(in_dim, out_dim, init_thres=init_thres, q_dim=low_rank_dim, norm_p=norm_p, enable_bias=False)
        self.drop_path = nn.Identity() if kwargs['no_drop_path'] else DropPath(0.1)

    def forward(self, x):
        return self.drop_path(self.learnable(x))


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0., tuning_mode=None, lora_bias=False, low_rank_dim=-1, tuning_model=None, init_thres=-0.1, **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

        self.additional_fc1 = TuningModule(in_features, hidden_features, low_rank_dim, tuning_model=tuning_model, bias=False, init_thres=init_thres)

        self.additional_fc2 = TuningModule(hidden_features, out_features, low_rank_dim, tuning_model=tuning_model, bias=False, init_thres=init_thres)
        self.additional_drop = nn.Dropout(p=0.1)

        self.tuning_mode = tuning_mode


    def forward(self, x):
        x_delta = 0.
        x_delta = self.additional_fc1(self.additional_drop(x))

        x = self.fc1(x) + x_delta

        x = self.act(x)
        x = self.drop1(x)

        x_delta = 0.
        x_delta = self.additional_fc2(self.additional_drop(x))

        x = self.fc2(x) + x_delta
        
        x = self.drop2(x)
        
        return x



class ConvNeXtBlock(nn.Module):
    """ ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., ls_init_value=1e-6, conv_mlp=False, mlp_ratio=4, norm_layer=None, tuning_mode=None, low_rank_dim=-1, tuning_model=None, init_thres=-0.1):
        super().__init__()
        if not norm_layer:
            norm_layer = partial(LayerNorm2d, eps=1e-6) if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        mlp_layer = ConvMlp if conv_mlp else Mlp
        self.use_conv_mlp = conv_mlp
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, tuning_mode=tuning_mode, low_rank_dim=low_rank_dim, tuning_model=tuning_model, init_thres=init_thres)

        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.tuning_mode = tuning_mode




    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)

        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)

            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut

        return x


class Downsample(nn.Module):
    """ 2D Image to Downsample
    """
    def __init__(self, dim, out_dim, kernel_size, stride, norm_layer=None, tuning_mode=None):
        super().__init__()

        self.norm = norm_layer(dim)
        self.proj = nn.Conv2d(dim, out_dim, kernel_size=stride, stride=stride)

        self.tuning_mode = tuning_mode


    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)

        return x
            


class ConvNeXtStage(nn.Module):

    def __init__(
            self, in_chs, out_chs, stride=2, depth=2, dp_rates=None, ls_init_value=1.0, conv_mlp=False,
            norm_layer=None, cl_norm_layer=None, cross_stage=False, tuning_mode=None, low_rank_dim=-1, tuning_model=None, init_thres=-0.1):
        super().__init__()
        self.grad_checkpointing = False 

        if in_chs != out_chs or stride > 1:
            self.downsample = Downsample(dim=in_chs, out_dim=out_chs, kernel_size=stride, stride=stride, norm_layer=norm_layer, tuning_mode=tuning_mode)
        else:
            self.downsample = nn.Identity()

        dp_rates = dp_rates or [0.] * depth
        self.blocks = nn.Sequential(*[ConvNeXtBlock(
            dim=out_chs, drop_path=dp_rates[j], ls_init_value=ls_init_value, conv_mlp=conv_mlp,
            norm_layer=norm_layer if conv_mlp else cl_norm_layer, tuning_mode=tuning_mode[j], low_rank_dim=low_rank_dim, tuning_model=tuning_model, init_thres=init_thres)
            for j in range(depth)]
        )

    def forward(self, x):
        x = self.downsample(x)

        #if self.grad_checkpointing and not torch.jit.is_scripting():
        #    x = checkpoint_seq(self.blocks, x)
        #else:
        x = self.blocks(x)

        return x



class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, tuning_mode=None):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.tuning_mode = tuning_mode


    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)

        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
            self, in_chans=3, num_classes=1000, global_pool='avg', output_stride=32, patch_size=4,
            depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),  ls_init_value=1e-6, conv_mlp=False,
            head_init_scale=1., head_norm_first=False, norm_layer=None, drop_rate=0., drop_path_rate=0., tuning_mode=None, low_rank_dim=-1, tuning_model=None, init_thres=-0.1, **kwargs
    ):
        super().__init__()
        assert output_stride == 32
        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)
            cl_norm_layer = norm_layer if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        else:
            assert conv_mlp,\
                'If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input'
            cl_norm_layer = norm_layer

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []

        # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
        self.stem = PatchEmbed(patch_size=4, in_chans=3, embed_dim=dims[0], norm_layer=norm_layer, tuning_mode=tuning_mode)

        self.tuning_mode = tuning_mode
        tuning_mode_list = [[tuning_mode] * depths[i_layer] for i_layer in range(len(depths))]

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        curr_stride = patch_size
        prev_chs = dims[0]
        stages = []
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            stride = 2 if i > 0 else 1
            # FIXME support dilation / output_stride
            curr_stride *= stride
            out_chs = dims[i]
            stages.append(ConvNeXtStage(
                prev_chs, out_chs, stride=stride,
                depth=depths[i], dp_rates=dp_rates[i], ls_init_value=ls_init_value, conv_mlp=conv_mlp,
                norm_layer=norm_layer, cl_norm_layer=cl_norm_layer, tuning_mode=tuning_mode_list[i], low_rank_dim=low_rank_dim, tuning_model=tuning_model, init_thres=init_thres)
            )
            prev_chs = out_chs
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)

        self.num_features = prev_chs
        if head_norm_first:
            # norm -> global pool -> fc ordering, like most other nets (not compat with FB weights)
            self.norm_pre = norm_layer(self.num_features)  # final norm layer, before pooling
            self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)
        else:
            # pool -> norm -> fc, the default ConvNeXt ordering (pretrained FB weights)
            self.norm_pre = nn.Identity()
            self.head = nn.Sequential(OrderedDict([
                ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
                ('norm', norm_layer(self.num_features)),
                ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
                ('drop', nn.Dropout(self.drop_rate)),
                ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
            ]))

        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)
        self.freeze_stages()
        #if weight_init != 'skip':
        #    self.init_weights(weight_init)

    def freeze_stages(self):

        self.stem.eval()

        fine_tune_keywords = ['head', 'cls_token', 'additional', 'sparse_', 'lora', 'adapter']
        #fine_tune_keywords.extend(self.fully_fine_tuned_keys)
        for name,param in self.named_parameters():
            if not any(kwd in name for kwd in fine_tune_keywords):
                param.requires_grad = False

        #if self.ft_cls_token:
        #    self.cls_token.requires_grad = True

        total_para_nums = 0
        head_para_nums = 0
        additional_para_nums = 0
        else_names = []
        additional_names = []

        for name, param in self.named_parameters():
            print(name, param.requires_grad)
            if param.requires_grad:

                if 'additional' in name:
                    additional_names.append(name)
                    additional_para_nums += param.numel()
                    total_para_nums += param.numel()

                elif 'head' in name:
                    head_para_nums += param.numel()

                else:
                    total_para_nums += param.numel()
                    else_names.append(name)

        print('parameters:', total_para_nums, 'head: ', head_para_nums)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool='avg'):
        if isinstance(self.head, ClassifierHead):
            # norm -> global pool -> fc
            self.head = ClassifierHead(
                self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)
        else:
            # pool -> norm -> fc
            self.head = nn.Sequential(OrderedDict([
                ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
                ('norm', self.head.norm),
                ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
                ('drop', nn.Dropout(self.drop_rate)),
                ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
            ]))

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)

        return x


    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        nn.init.constant_(module.bias, 0)
        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def checkpoint_filter_fn(state_dict, model):
    """ Remap FB checkpoints -> timm """
    #ipdb.set_trace()
    if 'model' in state_dict:
        state_dict = state_dict['model']
    out_dict = {}
    import re
    for k, v in state_dict.items():
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
        out_dict[k] = v
    return out_dict


def _create_convnext(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    #repr_size = kwargs.pop('representation_size', None)
    #if repr_size is not None and num_classes != default_num_classes:
    #    # Remove representation layer if fine-tuning. This may not always be the desired action,
    #    # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
    #    _logger.warning("Removing representation layer for fine-tuning.")
    #    repr_size = None

    model = build_model_with_cfg(
        ConvNeXt, variant, pretrained,
        default_cfg=default_cfg,
        #representation_size=repr_size,
        #representation_size=None,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs)
    return model

@register_model
def convnext_tiny(pretrained=False, **kwargs):
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)
    model = _create_convnext('convnext_tiny', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_tiny_hnf(pretrained=False, **kwargs):
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), head_norm_first=True, **kwargs)
    model = _create_convnext('convnext_tiny_hnf', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_small(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    model = _create_convnext('convnext_small', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_base(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model = _create_convnext('convnext_base', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_large(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    model = _create_convnext('convnext_large', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_base_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model = _create_convnext('convnext_base_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_large_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    model = _create_convnext('convnext_large_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_xlarge_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    model = _create_convnext('convnext_xlarge_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_base_384_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model = _create_convnext('convnext_base_384_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_large_384_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    model = _create_convnext('convnext_large_384_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_xlarge_384_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    model = _create_convnext('convnext_xlarge_384_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_base_in22k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model = _create_convnext('convnext_base_in22k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_large_in22k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    model = _create_convnext('convnext_large_in22k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_xlarge_in22k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    model = _create_convnext('convnext_xlarge_in22k', pretrained=pretrained, **model_args)
    return model
