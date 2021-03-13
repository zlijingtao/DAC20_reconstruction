#### Models for CIFAR-10 ############
from .unite_resnet_cifar import unite_resnet20, unite_resnet32, unite_resnet44, unite_resnet56

from .quan_resnet_cifar import quan_resnet20, quan_resnet32, quan_resnet44, quan_resnet56

from .quan_resnet_cifar_dual_nobn import dual_quan_resnet20_nobn, dual_quan_resnet32_nobn, dual_quan_resnet44_nobn, dual_quan_resnet56_nobn

from .quan_resnet_cifar_dual import dual_quan_resnet20, dual_quan_resnet32, dual_quan_resnet44, dual_quan_resnet56

from .quan_resnet_cifar_check import quan_resnet20_c, quan_resnet32_c, quan_resnet44_c, quan_resnet56_c

from .Bquan_resnet_cifar import bquan_resnet20, bquan_resnet32, bquan_resnet44, bquan_resnet56

from .vanilla_resnet_cifar import resnet20, resnet32, resnet44, resnet56
from .vanilla_resnet_cifar_act_3bit import resnet20_act_3bit, resnet32_act_3bit, resnet44_act_3bit, resnet56_act_3bit

from .tern_resnet_cifar import tern_resnet20, tern_resnet32, tern_resnet44, tern_resnet56
from .tern_resnet_cifar_act_3bit import tern_resnet20_act_3bit, tern_resnet32_act_3bit, tern_resnet44_act_3bit, tern_resnet56_act_3bit

from .bin_resnet_cifar import bin_resnet20, bin_resnet32, bin_resnet44, bin_resnet56
from .bin_resnet_cifar_nosave import bin_resnet20_ns, bin_resnet32_ns, bin_resnet44_ns, bin_resnet56_ns
from .bin_resnet_cifar_act_3bit import bin_resnet20_act_3bit, bin_resnet32_act_3bit, bin_resnet44_act_3bit, bin_resnet56_act_3bit
#### Models for MNIST ############
from .lenet5_bin import lenet5_bin
from .lenet5_tern import lenet5_tern
from .mlp_bin import mlp_bin
from .mlp_tern import mlp_tern
from .mlp_vanilla import mlp
from .mlp_512_bin import mlp_512_bin
from .mlp_512_tern import mlp_512_tern
from .mlp_512_vanilla import mlp_512
from .mlp_256_bin import mlp_256_bin
from .mlp_256_tern import mlp_256_tern
from .mlp_256_vanilla import mlp_256
#### Models for ImageNet ############
from .alexnet_vanilla import alexnet_vanilla
from .alexnet_quan import tern_alexnet_ff_lf, tern_alexnet_fq_lq

from .ResNet_quan import resnet18b_quan, resnet34b_quan, resnet50b_quan, resnet101b_quan

from .ResNet_tern import resnet18b_ff_lf_tex1, resnet18b_fq_lq_tex1
from .ResNet_tern import resnet34b_ff_lf_tex1, resnet34b_fq_lq_tex1
from .ResNet_tern import resnet50b_ff_lf_tex1, resnet50b_fq_lq_tex1
from .ResNet_tern import resnet101b_ff_lf_tex1, resnet101b_fq_lq_tex1

from .ResNet_tern_act_3bit import resnet18b_ff_lf_tex1_act_3bit, resnet18b_fq_lq_tex1_act_3bit
from .ResNet_tern_act_3bit import resnet34b_ff_lf_tex1_act_3bit, resnet34b_fq_lq_tex1_act_3bit
from .ResNet_tern_act_3bit import resnet50b_ff_lf_tex1_act_3bit, resnet50b_fq_lq_tex1_act_3bit
from .ResNet_tern_act_3bit import resnet101b_ff_lf_tex1_act_3bit, resnet101b_fq_lq_tex1_act_3bit

from .ResNet_bin import resnet18b_ff_lf_bin, resnet18b_fq_lq_bin
from .ResNet_bin import resnet34b_ff_lf_bin, resnet34b_fq_lq_bin
from .ResNet_bin import resnet50b_ff_lf_bin, resnet50b_fq_lq_bin
from .ResNet_bin import resnet101b_ff_lf_bin, resnet101b_fq_lq_bin

from .ResNet_bin_act_3bit import resnet18b_ff_lf_bin_act_3bit, resnet18b_fq_lq_bin_act_3bit
from .ResNet_bin_act_3bit import resnet34b_ff_lf_bin_act_3bit, resnet34b_fq_lq_bin_act_3bit
from .ResNet_bin_act_3bit import resnet50b_ff_lf_bin_act_3bit, resnet50b_fq_lq_bin_act_3bit
from .ResNet_bin_act_3bit import resnet101b_ff_lf_bin_act_3bit, resnet101b_fq_lq_bin_act_3bit

from .ResNet_REL_tex2 import resnet18b_fq_lq_tern_tex_2
from .ResNet_REL_tex4 import resnet18b_fq_lq_tern_tex_4

from .resnet_vanilla import resnet18, resnet34, resnet50, resnet101, resnet152
from .resnet_vanilla_act_3bit import resnet18_act_3bit, resnet34_act_3bit, resnet50_act_3bit, resnet101_act_3bit, resnet152_act_3bit