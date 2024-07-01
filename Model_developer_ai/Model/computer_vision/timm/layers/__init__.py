import sys
from pathlib import Path
current_file = Path(__file__).resolve()
package_root = current_file.parents[2]
sys.path.append(str(package_root))
# print(f"Running {current_file.name} with PYTHONPATH set to {package_root}")

from timm.layers.activations import *
from timm.layers.adaptive_avgmax_pool import \
      adaptive_avgmax_pool2d, select_adaptive_pool2d, AdaptiveAvgMaxPool2d, SelectAdaptivePool2d
from timm.layers.attention_pool import AttentionPoolLatent
from timm.layers.attention_pool2d import AttentionPool2d, RotAttentionPool2d, RotaryEmbedding
from timm.layers.blur_pool import BlurPool2d
from timm.layers.classifier import ClassifierHead, create_classifier, NormMlpClassifierHead
from timm.layers.cond_conv2d import CondConv2d, get_condconv_initializer
from timm.layers.config import is_exportable, is_scriptable, is_no_jit, use_fused_attn, \
    set_exportable, set_scriptable, set_no_jit, set_layer_config, set_fused_attn
from timm.layers.conv2d_same import Conv2dSame, conv2d_same
from timm.layers.conv_bn_act import ConvNormAct, ConvNormActAa, ConvBnAct
from timm.layers.create_act import create_act_layer, get_act_layer, get_act_fn
from timm.layers.create_attn import get_attn, create_attn
from timm.layers.create_conv2d import create_conv2d
from timm.layers.create_norm import get_norm_layer, create_norm_layer
from timm.layers.create_norm_act import get_norm_act_layer, create_norm_act_layer, get_norm_act_layer
from timm.layers.drop import DropBlock2d, DropPath, drop_block_2d, drop_path
from timm.layers.eca import EcaModule, CecaModule, EfficientChannelAttn, CircularEfficientChannelAttn
from timm.layers.evo_norm import EvoNorm2dB0, EvoNorm2dB1, EvoNorm2dB2,\
    EvoNorm2dS0, EvoNorm2dS0a, EvoNorm2dS1, EvoNorm2dS1a, EvoNorm2dS2, EvoNorm2dS2a
from timm.layers.fast_norm import is_fast_norm, set_fast_norm, fast_group_norm, fast_layer_norm
from timm.layers.filter_response_norm import FilterResponseNormTlu2d, FilterResponseNormAct2d
from timm.layers.format import Format, get_channel_dim, get_spatial_dim, nchw_to, nhwc_to
from timm.layers.gather_excite import GatherExcite
from timm.layers.global_context import GlobalContext
from timm.layers.grid import ndgrid, meshgrid
from timm.layers.helpers import to_ntuple, to_2tuple, to_3tuple, to_4tuple, make_divisible, extend_tuple
from timm.layers.inplace_abn import InplaceAbn
from timm.layers.linear import Linear
from timm.layers.mixed_conv2d import MixedConv2d
from timm.layers.mlp import *
# from timm.layers.mlp import Mlp, GluMlp, GatedMlp, SwiGLU, SwiGLUPacked, ConvMlp,GlobalResponseNormMlp
from timm.layers.non_local_attn import NonLocalAttn, BatNonLocalAttn
from timm.layers.norm import GroupNorm, GroupNorm1, LayerNorm, LayerNorm2d, RmsNorm
from timm.layers.norm_act import BatchNormAct2d, GroupNormAct, GroupNorm1Act, LayerNormAct, LayerNormAct2d,\
    SyncBatchNormAct, convert_sync_batchnorm, FrozenBatchNormAct2d, freeze_batch_norm_2d, unfreeze_batch_norm_2d
from timm.layers.padding import get_padding, get_same_padding, pad_same
from timm.layers.patch_dropout import PatchDropout
from timm.layers.patch_embed import PatchEmbed, PatchEmbedWithSize, resample_patch_embed
from timm.layers.pool2d_same import AvgPool2dSame, create_pool2d
from timm.layers.pos_embed import resample_abs_pos_embed, resample_abs_pos_embed_nhwc
from timm.layers.pos_embed_rel import RelPosMlp, RelPosBias, RelPosBiasTf, gen_relative_position_index, gen_relative_log_coords, \
    resize_rel_pos_bias_table, resize_rel_pos_bias_table_simple, resize_rel_pos_bias_table_levit
from timm.layers.pos_embed_sincos import pixel_freq_bands, freq_bands, build_sincos2d_pos_embed, build_fourier_pos_embed, \
    build_rotary_pos_embed, apply_rot_embed, apply_rot_embed_cat, apply_rot_embed_list, apply_keep_indices_nlc, \
    FourierEmbed, RotaryEmbedding, RotaryEmbeddingCat
from timm.layers.squeeze_excite import SEModule, SqueezeExcite, EffectiveSEModule, EffectiveSqueezeExcite
from timm.layers.selective_kernel import SelectiveKernel
from timm.layers.separable_conv import SeparableConv2d, SeparableConvNormAct
from timm.layers.space_to_depth import SpaceToDepth, DepthToSpace
from timm.layers.split_attn import SplitAttn
from timm.layers.split_batchnorm import SplitBatchNorm2d, convert_splitbn_model
from timm.layers.std_conv import StdConv2d, StdConv2dSame, ScaledStdConv2d, ScaledStdConv2dSame
from timm.layers.test_time_pool import TestTimePoolHead, apply_test_time_pool
from timm.layers.trace_utils import _assert, _float_to_int
from timm.layers.typing1 import LayerType, PadType
from timm.layers.weight_init import trunc_normal_, trunc_normal_tf_, variance_scaling_, lecun_normal_
from timm.layers.grn import  GlobalResponseNorm