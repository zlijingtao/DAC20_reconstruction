from __future__ import division
from __future__ import absolute_import

import os
import sys
import shutil
import time
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchsummary import summary
import statistics
# import torchvision.utils.make_grid as make_grid
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
from tensorboardX import SummaryWriter
from utils_.reorganize_param import reorganize_param
# from tensorboardX import SummaryWriter
import models

from models.codebook import get_code_book
# from models.quan_resnet_cifar import quan_Conv2d, quan_Linear
from inspect import signature

# import yellowFin tuner
sys.path.append("./tuner_utils")
from tuner_utils.yellowfin import YFOptimizer

from attack.BFA import *
import torch.nn.functional as F
import copy
from models.fixpoint_modules import  get_centroid, get_quantized, quantize
from models.check_modules import get_qcode
import config
writer = SummaryWriter('runs/2x/weight_data_0')
# writer.flush()
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

################# Options ##################################################
############################################################################
parser = argparse.ArgumentParser(description='Training network for image classification',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', default='/home/elliot/data/pytorch/svhn/',
                    type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist'],
                    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='lbcnn', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', dest='evaluate',
                    action='store_true', help='evaluate model on validation set')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
#Checker Parameters
parser.add_argument('--check', dest='enable_check', action='store_true',
                    help='enable weight integrity checking')
parser.add_argument('--massive', type=int, default=100, help='Number of massive test')
parser.add_argument('--check_gsize', type=int, default=4, help='Number of elements per group')
parser.add_argument('--check_factor', type=float, default=4.0, help='factor on sigma on quantization range')
parser.add_argument('--check_bit', type=int, default=2, help='Number of bit in quantizing the coded mean')
parser.add_argument('--limit_row', type=int, default=10, help='Limitation on number of rows of codebook')
# Hybrid Scheme Related Parameters
parser.add_argument('--input_grain_size', nargs='+', type=int, default=[1, 1], help='Grain size to calculate Mean in input layer')
parser.add_argument('--input_num_bits', type=int, default=4, help='Number of bits for Mean representation in input layer')
parser.add_argument('--input_M2D', type=float, default=0.0, help='Mean-to-Deviation ratio in input layer')
parser.add_argument('--res_grain_size', nargs='+', type=int, default=[1, 1], help='Grain size to calculate Mean in resnet block conv layer')
parser.add_argument('--res_num_bits', type=int, default=4, help='Number of bits for Mean representation in resnet block conv layer')
parser.add_argument('--res_M2D', type=float, default=0.0, help='Mean-to-Deviation ratio in resnet block conv layer')
parser.add_argument('--output_grain_size', nargs='+', type=int, default=[1, 1], help='Grain size to calculate Mean in output layer')
parser.add_argument('--output_num_bits', type=int, default=4, help='Number of bits for Mean representation in output layer')
parser.add_argument('--output_M2D', type=float, default=0.0, help='Mean-to-Deviation ratio in output layer')
# AD NOISE (Only used in testing the model)
parser.add_argument('--AD_sigma', type=float, default=0.0, help='AD_sigma')
parser.add_argument('--DA_sigma', type=float, default=0.0, help='DA_sigma')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='device range [0,ngpu-1]')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, default=5000, help='manual seed')
# quantization
parser.add_argument('--reset_weight', dest='reset_weight', action='store_true',
                    help='enable the weight replacement with the quantized weight')
# Bit Flip Attacked
parser.add_argument('--layer_id', type=int, default=0,
                    help='indicate which layer to attack, default to 0, attack every layer with the same rate.')
parser.add_argument('--bfa', dest='enable_bfa', action='store_true',
                    help='enable the bit-flip attack')
parser.add_argument('--oneshot', dest='enable_oneshot', action='store_true',
                    help='enable the oneshot attack')
parser.add_argument('--rfa', dest='enable_rfa', action='store_true',
                    help='enable the random bit-flip attack')
parser.add_argument('--wma', dest='enable_wma', action='store_true',
                    help='enable the weight manipulation attack')
parser.add_argument('--attack_sample_size', type=int, default=128,
                    help='attack sample size')
parser.add_argument('--n_iter', type=int, default=20,
                    help='number of attack iterations')
parser.add_argument( '--k_top', type=int, default=10,
                    help='IF BFA: k weight with top ranking gradient used for bit-level gradient check; IF ONESHOT/RFA: k gradient-based bit flips per layer'
)
parser.add_argument('--update_mask_flag', dest='update_mask_flag', action='store_true',
                    help='enable RHA constraint BFA')
##########################################################################

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.ngpu == 1:
    # make only device #gpu_id visible, then
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check GPU

# Give a random seed if no manual configuration
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True


###############################################################################
###############################################################################

def main():
    log = open(os.path.join(os.path.dirname(args.resume), 'test_log.txt'), 'w')
    print_log("Evaluate saved Model : {}".format(args.resume), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("python version : {}".format(
        sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(
        torch.backends.cudnn.version()), log)

    # Init the tensorboard path and writer
    # tb_path = os.path.join(args.resume, 'tb_log',
                           # 'run_' + str(args.manualSeed))
    # logger = Logger(tb_path)
    # writer = SummaryWriter(tb_path)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.dataset == 'svhn':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'mnist':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    if args.dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    if args.dataset == 'mnist':
        train_data = dset.MNIST(args.data_path, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ]))
        test_data = dset.MNIST(args.data_path, train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ]))
        num_classes = 10
    elif args.dataset == 'cifar10':
        train_data = dset.CIFAR10(
            args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(
            args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(
            args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(
            args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path, split='train',
                               transform=train_transform, download=True)
        test_data = dset.SVHN(args.data_path, split='test',
                              transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(
            args.data_path, split='train', transform=train_transform, download=True)
        test_data = dset.STL10(args.data_path, split='test',
                               transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val')
        train_data = dset.ImageFolder(train_dir, transform=train_transform)
        test_data = dset.ImageFolder(test_dir, transform=test_transform)
        num_classes = 1000
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    
    # Model Creating
    print_log("=> creating model '{}'".format(args.arch), log)
    model_param_dict = signature(models.__dict__[args.arch]).parameters
    # Init model, criterion, and optimizer
    if ('AD_sigma' in model_param_dict) and ('input_grain_size' in model_param_dict):
        net = models.__dict__[args.arch](num_classes, args.AD_sigma, args.DA_sigma, args.input_grain_size, args.input_num_bits, args.input_M2D, args.res_grain_size, args.res_num_bits, args.res_M2D, args.output_grain_size, args.output_num_bits, args.output_M2D, os.path.dirname(args.resume))
    elif 'input_grain_size' in model_param_dict:
        net = models.__dict__[args.arch](num_classes, args.input_grain_size, args.input_num_bits, args.input_M2D, args.res_grain_size, args.res_num_bits, args.res_M2D, args.output_grain_size, args.output_num_bits, args.output_M2D, os.path.dirname(args.resume))
    elif 'AD_sigma' in model_param_dict:
        net = models.__dict__[args.arch](num_classes, args.AD_sigma, args.DA_sigma)
    else:
        net = models.__dict__[args.arch](num_classes)
    if args.use_cuda:
        if args.ngpu > 1:
            net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))


    criterion = torch.nn.CrossEntropyLoss()
    if args.use_cuda:
        net.cuda()
        criterion.cuda()
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)

            state_tmp = net.state_dict()
            if 'state_dict' in checkpoint.keys():
                state_tmp.update(checkpoint['state_dict'])
            else:
                state_tmp.update(checkpoint)

            net.load_state_dict(state_tmp)
            # net.load_state_dict(checkpoint['state_dict'])

            print_log("=> loaded checkpoint '{}'".format(
                args.resume), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log(
            "=> do not use any checkpoint for {} model".format(args.arch), log)
    if args.reset_weight:
        for name, m in net.named_modules():
            if ("conv" in name) or ("classifier" in name) or ("fc" in name) or ("downsample.0" in name):
                print(m)
                m.__reset_weight__()

    if args.enable_check:

        print_log("=> Generating Codebook for checking group size '{}'".format(args.check_gsize), log)
        codebook = get_code_book([1, args.check_gsize], args.limit_row)
        print(models.__dict__[args.arch+"_c"])
        print_log("=> creating model '{}'".format(args.arch+"_c"), log)
        model_param_dict_c = signature(models.__dict__[args.arch+"_c"]).parameters
        # print(num_classes)
        # Init model, criterion, and optimizer
        if ('AD_sigma' in model_param_dict_c) and ('input_grain_size' in model_param_dict_c):
            net_c = models.__dict__[args.arch+"_c"](codebook, num_classes, args.AD_sigma, args.DA_sigma, args.input_grain_size, args.input_num_bits, args.input_M2D, args.res_grain_size, args.res_num_bits, args.res_M2D, args.output_grain_size, args.output_num_bits, args.output_M2D, os.path.dirname(args.resume), args.check_gsize, args.check_factor, args.check_bit)
        elif 'input_grain_size' in model_param_dict_c:
            net_c = models.__dict__[args.arch+"_c"](codebook, num_classes, args.input_grain_size, args.input_num_bits, args.input_M2D, args.res_grain_size, args.res_num_bits, args.res_M2D, args.output_grain_size, args.output_num_bits, args.output_M2D, os.path.dirname(args.resume), args.check_gsize, args.check_factor, args.check_bit)
        elif 'AD_sigma' in model_param_dict_c:
            net_c = models.__dict__[args.arch+"_c"](codebook, num_classes, args.AD_sigma, args.DA_sigma, args.check_gsize, args.check_factor, args.check_bit)
        else:
            net_c = models.__dict__[args.arch+"_c"](codebook, num_classes, args.check_gsize, args.check_factor, args.check_bit)
        if args.use_cuda:
            if args.ngpu > 1:
                net_c = torch.nn.DataParallel(net_c, device_ids=list(range(args.ngpu)))
            net_c.cuda()
        if args.resume:
            if os.path.isfile(args.resume):
                print_log("=> loading checkpoint '{}'".format(args.resume), log)
                checkpoint = torch.load(args.resume)
                state_tmp = net_c.state_dict()
                if 'state_dict' in checkpoint.keys():
                    state_tmp.update(checkpoint['state_dict'])
                else:
                    state_tmp.update(checkpoint)

                net_c.load_state_dict(state_tmp)
                # net_c.load_state_dict(checkpoint['state_dict'])

                print_log("=> loaded checkpoint for the checker model '{}'".format(
                    args.resume), log)
            else:
                print_log("=> no checkpoint found at '{}'".format(args.resume), log)
        else:
            print_log(
                "=> do not use any checkpoint for {} model".format(args.arch), log)
        if args.reset_weight:
            for name, m in net_c.named_modules():
                if ("conv" in name) or ("classifier" in name) or ("fc" in name) or ("downsample.0" in name):
                    m.__reset_weight__()
    # counter = 0
    # for name, module in net.named_modules():
        # if ("conv" in name) or ("classifier" in name):
            # counter += 1
            # weight_data = module.weight.data.cpu().numpy()
            # weight_data  = weight_data.flatten()
            # plt.clf()
            # plt.figure(figsize=(12, 9))
            # ax = plt.subplot(111)
            # ax.spines["top"].set_visible(False)
            # ax.spines["right"].set_visible(False)
            # ax.get_xaxis().tick_bottom()
            # ax.get_yaxis().tick_left()
            # plt.xticks(fontsize=30)
            # plt.yticks(fontsize=30)
            # plt.xlabel("Weight Value", fontsize=32)
            # plt.ylabel("Count", fontsize=32)
            # plt.hist(weight_data, color="#3F5D7D", bins = 64)
            # plt.savefig("./weight_histo_{}.pdf".format(counter), bbox_inches="tight")
    summary(net, (3, 32, 32))
    attacker = BFA(criterion, args.k_top)
    if args.enable_check:
        net_clean = copy.deepcopy(net_c)
    else:
        net_c = copy.deepcopy(net)
        net_clean = copy.deepcopy(net_c)
    net_dirty = copy.deepcopy(net)
    if args.enable_bfa and args.enable_oneshot:
        print("Warning! Both bit flip activated, should only activate one at a time.")
        return
    if args.enable_bfa or args.enable_oneshot or args.enable_rfa:
        # args.manualSeed = 25
        args.manualSeed = 125
        print_log( "=> Seed is {}".format(args.manualSeed), log)
        random.seed(args.manualSeed)
        torch.manual_seed(args.manualSeed)

    if args.use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
    if args.enable_bfa:
        founded = []
        pass_all = 0
        orig_accu = 0
        recovered_accu = 0
        attacker = BFA(criterion, args.k_top, layer_id = args.layer_id)
        path_to_sinature = os.path.dirname(args.resume)+ '/saved_tensors/'
        for fname in os.listdir(path_to_sinature):
            if fname.startswith("signature"):
                # print(fname)
                os.remove(os.path.join(path_to_sinature, fname))
        orig_accu_top1, _, _ = validate(test_loader, net_clean, criterion, log)
        print_log( "=> Original Accuracy is {}%".format(orig_accu_top1), log)
        config.check = 0
        net_real_clean =  copy.deepcopy(net_c)
        for i in range(args.massive):
            net_clean = copy.deepcopy(net_c)
            net_dirty = copy.deepcopy(net)
            print("Massive Test: ({}/{})".format(i+1, args.massive))
            if args.layer_id != 0:
                accu, loss, grad, accu_orig = perform_attack(attacker, net_dirty, net_clean, net_real_clean, args.enable_check, train_loader, test_loader, args.n_iter, log, args.update_mask_flag, True, False, False)
                if i == 0:
                    run_grad = grad
                else:
                    run_grad = run_grad + grad
            else:
                accu, loss, grad, accu_orig = perform_attack(attacker, net_dirty, net_clean, net_real_clean, args.enable_check, train_loader, test_loader, args.n_iter, log, args.update_mask_flag, False, False, False)
            # print_log(attacker.attack_bit, log)
            # print_log(attacker.attack_layer, log)

            for key1 in attacker.attack_index.copy():
                for key2 in attacker.attack_index[key1].copy():
                    if attacker.attack_index[key1][key2] == 1:
                        del attacker.attack_index[key1][key2]

            for key1 in attacker.attack_index.copy():
                if not attacker.attack_index[key1]:
                    attacker.attack_index.pop(key1)
            # print_log(attacker.attack_index, log)
            # print_log(attacker.attack_weight_value, log)

            attacker.attack_index = {}
            orig_accu += accu_orig
            if args.enable_check:
                pass_all += (config.check == 0)
                founded.append(config.check)
                recovered_accu += accu
                if config.check == 0:
                    print_log("Model Integrity Check Passed!", log)
                else:
                    print_log("Model Integrity Check Failed!, we found {} bit-flips".format(config.check), log)
                config.check = 0
        if args.layer_id != 0:
            gradient_map = run_grad
            # print(gradient_map.shape)
            if len(gradient_map.shape) == 4:
                gradient_map = gradient_map.reshape(gradient_map.shape[0], -1)
            gradient_map_normed = (gradient_map - np.mean(gradient_map))/np.var(gradient_map)
            plt.imshow(gradient_map_normed, cmap='hot', interpolation='nearest')
            plt.savefig("./gradient_map_layer{}_seed{}_avg_{}.png".format(args.layer_id,args.manualSeed, args.massive))

    if args.enable_oneshot:
        founded = []
        pass_all = 0
        pass_accu = 0
        fail_accu = 0
        attacker = oneshot_BFA(criterion, args.k_top, layer_id = args.layer_id)
        worst_pass_accu = 100.0
        worst_fail_accu = 100.0
        path_to_sinature = os.path.dirname(args.resume)+ '/saved_tensors/'
        for fname in os.listdir(path_to_sinature):
            if fname.startswith("signature"):
                # print(fname)
                os.remove(os.path.join(path_to_sinature, fname))
        orig_accu_top1, _, _ = validate(test_loader, net_clean, criterion, log)
        print_log( "=> Original Accuracy is {}%".format(orig_accu_top1), log)
        config.check = 0
        for i in range(args.massive):
            net_clean = copy.deepcopy(net_c)
            net_dirty = copy.deepcopy(net)
            print("Massive Test: ({}/{})".format(i+1, args.massive))
            accu, loss = perform_oneshotattack(attacker, net_dirty, net_clean, train_loader, test_loader, log)
            pass_all += (config.check == 0)
            founded.append(config.check)

            if config.check == 0:
                pass_accu += accu
                if accu <= worst_pass_accu:
                    worst_pass_accu = accu
                print_log("Model Integrity Check Passed!", log)
            else:
                fail_accu += accu
                if accu <= worst_fail_accu:
                    worst_fail_accu = accu
                print_log("Model Integrity Check Failed!", log)
            config.check = 0

    if args.enable_rfa:
        founded = []
        pass_all = 0
        pass_accu = 0
        fail_accu = 0
        attacker = RFA(criterion, args.k_top, layer_id = args.layer_id)
        worst_pass_accu = 100.0
        worst_fail_accu = 100.0
        path_to_sinature = os.path.dirname(args.resume)+ '/saved_tensors/'
        for fname in os.listdir(path_to_sinature):
            if fname.startswith("signature"):
                # print(fname)
                os.remove(os.path.join(path_to_sinature, fname))
        orig_accu_top1, _, _ = validate(test_loader, net_clean, criterion, log)
        print_log( "=> Original Accuracy is {}%".format(orig_accu_top1), log)
        config.check = 0
        for i in range(args.massive):
            net_clean = copy.deepcopy(net_c)
            net_dirty = copy.deepcopy(net)
            print("Massive Test: ({}/{})".format(i+1, args.massive))
            accu, loss = perform_randomattack(attacker, net_dirty, net_clean, train_loader, test_loader, log)
            pass_all += (config.check == 0)
            founded.append(config.check)

            if config.check == 0:
                pass_accu += accu
                if accu <= worst_pass_accu:
                    worst_pass_accu = accu
                print_log("Model Integrity Check Passed!", log)
            else:
                fail_accu += accu
                if accu <= worst_fail_accu:
                    worst_fail_accu = accu
                print_log("Model Integrity Check Failed!", log)
            config.check = 0

    if args.enable_wma:
        pass_all = 0
        orig_accu = 0
        recovered_accu = 0
        attacker = weight_shifter(criterion, args.k_top, layer_id = args.layer_id)
        path_to_sinature = os.path.dirname(args.resume)+ '/saved_tensors/'
        for fname in os.listdir(path_to_sinature):
            if fname.startswith("signature"):
                # print(fname)
                os.remove(os.path.join(path_to_sinature, fname))
        orig_accu_top1, _, _ = validate(test_loader, net_clean, criterion, log)
        print_log( "=> Original Accuracy is {}%".format(orig_accu_top1), log)
        config.check = 0
        for i in range(args.massive):
            net_clean = copy.deepcopy(net_c)
            net_dirty = copy.deepcopy(net)
            print("Massive Test: ({}/{})".format(i+1, args.massive))
            accu, loss, pass_i, accu_orig = perform_weight_shifting_attack(attacker, net_dirty, net_clean, train_loader, test_loader, log)
            pass_all += (config.check == 0)
            founded.append(config.check)

            if config.check == 0:
                orig_accu += accu_orig
                recovered_accu += accu
                print_log("Model Integrity Check Passed!", log)
            else:
                orig_accu += accu_orig
                recovered_accu += accu
                print_log("Model Integrity Check Failed!", log)
            config.check = 0


    if args.enable_bfa or args.enable_oneshot or args.enable_rfa or args.enable_wma:
        # founded
        print_log( "=> Average Attacked Accuracy is {}%".format(orig_accu/args.massive), log)
        if args.enable_check:
            mean_founded = sum(founded) / len(founded)
            std_founded = statistics.stdev(founded)
            print_log( "=> Average detected bit-flips is {} and {}".format(mean_founded, std_founded), log)
        print_log("Massive Test on data integrity Finished!",log)
        return
    # time_cost = []
    for epoch in range(1):
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        val_acc, _, val_los = validate(test_loader, net, criterion, log)
        # end.record()
        # torch.cuda.synchronize()
        # time_cost.append(start.elapsed_time(end))
        # print(start.elapsed_time(end))
        # end = time.time()
        # val_acc, _, val_los = validate(test_loader, net, criterion, log)
        # time_cost.append(time.time() - end)
    # print('Time cost is %.8f' %(np.mean(time_cost)))
    log.close()
    print("Test Finished!")
    
'''Function to do testing, twin: add one dummy bit near the attacking bit to bypass the checker, lessen: pseudo-attacking the MSB-1'''
'''quick_tect, if True, do detection only without recovery(testing); skip_test: if True, do BFA without testing the degradation. They are good for detection robustness simulation'''
def perform_attack(attacker, model, model_check, model_clean, enable_check, train_loader, test_loader,
                   N_iter, log, update_mask_flag = False, save_stats = False, skip_test = False, quick_detect = False, twin = False, lessen = False):
    # Note that, attack has to be done in evaluation model due to batch-norm.
    # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
    model.eval()
    losses = AverageMeter()
    iter_time = AverageMeter()
    attack_time = AverageMeter()
    # attempt to use the training data to conduct BFA
    for _, (data, target) in enumerate(train_loader):
        if args.use_cuda:
            target = target.cuda(async=True)
            data = data.cuda()
        # Override the target to prevent label leaking
        temp1 = model(data)
        _, target = temp1.data.max(1)
        break

    print_log('k_top is set to {}'.format(args.k_top), log)
    print_log('Attack sample size is {}'.format(data.size()[0]), log)
    end = time.time()

    for i_iter in range(N_iter):
        print_log('**********************************', log)
        attack_index, attack_name, attack_gradient = attacker.progressive_bit_search(model, data, target, update_mask_flag, save_stats)
        # Print gradient heat map
        if save_stats:
            if i_iter == 0:
                gradient_map = config.grad_map
        else:
            gradient_map = None
        # measure data loading time
        attack_time.update(time.time() - end)
        end = time.time()

        # record the loss
        losses.update(attacker.loss_max, data.size(0))
        print_log(
            'Iteration: [{:03d}/{:03d}]   '
            'Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f})  '.
            format((i_iter + 1),
                   N_iter,
                   attack_time=attack_time,
                   iter_time=iter_time) + time_string(), log)

        print_log('loss before attack: {:.4f}'.format(attacker.loss.item()),
                  log)
        print_log('loss after attack: {:.4f}'.format(attacker.loss_max), log)

        print_log('bit flips: {:.0f}'.format(attacker.bit_counter), log)
        if i_iter == range(N_iter -1):
            attacker.bit_counter = 0
        if skip_test == True:
            val_acc_top1 = 1.0
            val_acc_top5 = 1.0
            val_loss = 0.0
            centroid_BF = {}
            deviation_BF = {}
        else:
            val_acc_top1, val_acc_top5, val_loss, centroid_BF, deviation_BF = validate_new(test_loader, model, attacker.criterion, log)

        iter_time.update(time.time() - end)
        print_log(
            'iteration Time {iter_time.val:.3f} ({iter_time.avg:.3f})'.format(
                iter_time=iter_time), log)
        end = time.time()

        if i_iter == N_iter - 1:
            print_log("End of searching, dump all the bits to the target system and perform the attack", log)

            for name, module in model.named_modules():
                if ("conv" in name) or ("classifier" in name) or ("fc" in name) or ("downsample.0" in name):
                    attack_weight = module.weight.data
                    for name2, module2 in model_clean.named_modules():
                        if name2 == name:
                            clean_weight = module2.weight.data
                    if twin == True:
                        if len(attack_weight.size()) == 4:
                            original_size = attack_weight.size()
                            reshaped_clean_weight = clean_weight.permute(1, 2, 3, 0)
                            reshaped_attack_weight = attack_weight.permute(1, 2, 3, 0)
                            initial_size = reshaped_attack_weight.size()
                            reshaped_clean_weight = reshaped_clean_weight.view(-1, original_size[0])
                            reshaped_attack_weight = reshaped_attack_weight.view(-1, original_size[0])
                            reshaped_clean_weight = reshaped_clean_weight.flatten()
                            reshaped_attack_weight = reshaped_attack_weight.flatten()
                        else:
                            original_size = attack_weight.size()
                            reshaped_clean_weight = clean_weight.view(original_size[0], original_size[1])
                            reshaped_attack_weight = attack_weight.view(original_size[0], original_size[1])
                            reshaped_clean_weight = reshaped_clean_weight.flatten()
                            reshaped_attack_weight = reshaped_attack_weight.flatten()
                        nonzero_index_array = (reshaped_attack_weight - reshaped_clean_weight).nonzero()
                        if nonzero_index_array.nelement() != 0:
                            for i in range(nonzero_index_array.size()[-1]):
                                nonzero_index = nonzero_index_array[i]
                                neighbor_index = nonzero_index + 1
                                if nonzero_index.item() == 0:
                                    neighbor_index = neighbor_index
                                elif (neighbor_index.item()/4) != (nonzero_index.item()/4):
                                    neighbor_index = nonzero_index - 1
                                changes = (reshaped_attack_weight - reshaped_clean_weight)[nonzero_index]
                                if reshaped_attack_weight[neighbor_index] > 0:
                                    reshaped_attack_weight[neighbor_index] -= 128
                                else:
                                    reshaped_attack_weight[neighbor_index] += 128
                                if len(attack_weight.size()) == 4:
                                    attack_weight = reshaped_attack_weight.view(-1, original_size[0]).view(initial_size).permute(3, 0, 1, 2)
                                else:
                                    attack_weight = reshaped_attack_weight.view(original_size[0], original_size[1]).view(original_size)
                        module.weight.data = attack_weight
                    if lessen == True:
                        if len(attack_weight.size()) == 4:
                            original_size = attack_weight.size()
                            reshaped_clean_weight = clean_weight.permute(1, 2, 3, 0)
                            reshaped_attack_weight = attack_weight.permute(1, 2, 3, 0)
                            initial_size = reshaped_attack_weight.size()
                            reshaped_clean_weight = reshaped_clean_weight.view(-1, original_size[0])
                            reshaped_attack_weight = reshaped_attack_weight.view(-1, original_size[0])
                            reshaped_clean_weight = reshaped_clean_weight.flatten()
                            reshaped_attack_weight = reshaped_attack_weight.flatten()
                        else:
                            original_size = attack_weight.size()
                            reshaped_clean_weight = clean_weight.view(original_size[0], original_size[1])
                            reshaped_attack_weight = attack_weight.view(original_size[0], original_size[1])
                            reshaped_clean_weight = reshaped_clean_weight.flatten()
                            reshaped_attack_weight = reshaped_attack_weight.flatten()
                        nonzero_index_array = (reshaped_attack_weight - reshaped_clean_weight).nonzero()
                        if nonzero_index_array.nelement() != 0:
                            for i in range(nonzero_index_array.size()[0]):
                                nonzero_index = nonzero_index_array[i]
                                # print(nonzero_index)
                                neighbor_index = nonzero_index
                                changes = (reshaped_attack_weight - reshaped_clean_weight)[nonzero_index]
                                # print(reshaped_clean_weight[nonzero_index])
                                if reshaped_attack_weight[neighbor_index] > 0:
                                    reshaped_attack_weight[neighbor_index] -= 64
                                elif reshaped_attack_weight[neighbor_index] < 0:
                                    reshaped_attack_weight[neighbor_index] += 64
                                # if reshaped_attack_weight[neighbor_index] > 32:
                                    # reshaped_attack_weight[neighbor_index] -= 160
                                # elif reshaped_attack_weight[neighbor_index] < -32:
                                    # reshaped_attack_weight[neighbor_index] += 160
                                # elif reshaped_attack_weight[neighbor_index] < 0:
                                     # reshaped_attack_weight[neighbor_index] += 96
                                # elif reshaped_attack_weight[neighbor_index] > 0:
                                     # reshaped_attack_weight[neighbor_index] -= 96

                                # print(reshaped_attack_weight[nonzero_index])
                                if len(attack_weight.size()) == 4:
                                    attack_weight = reshaped_attack_weight.view(-1, original_size[0]).view(initial_size).permute(3, 0, 1, 2)
                                else:
                                    attack_weight = reshaped_attack_weight.view(original_size[0], original_size[1]).view(original_size)
                        module.weight.data = attack_weight
            print_log("Accuracy after PBFA attack:", log)
            val_acc_top1_no_recover, val_acc_top5, val_loss, centroid_BF, deviation_BF = validate_new(test_loader, model, attacker.criterion, log)
            if enable_check:
                model_check.load_state_dict(model.state_dict())
                if quick_detect == True:
                    val_acc_top1 = 1.0
                    val_acc_top5 = 1.0
                    val_loss = 0.0
                    centroid_BF = {}
                    deviation_BF = {}
                    quick_detection(test_loader, model_check, attacker.criterion, log)
                else:
                    print_log("Perform zeroing recovery:", log)
                    val_acc_top1, val_acc_top5, val_loss, centroid_BF, deviation_BF = validate_new(test_loader, model_check, attacker.criterion, log)
            else:
                val_acc_top1 = 0
        # Check centroid_BF change
        # posit = 0
        # for key in centroid_BF:
            # posit += 1
            # if key == attack_name:
                # print_log("Attacking which layer: {:02d}".format(posit), log)
                # break

        # for key in centroid_BF:
            # if key == attack_name:
                # if len(centroid_CL[key].size()) == 4:
                    # centroid_CL_list = centroid_CL[key].permute(1, 2, 3, 0).contiguous().view(-1)
                    # centroid_BF_list = centroid_BF[key].permute(1, 2, 3, 0).contiguous().view(-1)
                    # deviation_CL_list = deviation_CL[key].permute(1, 2, 3, 0).contiguous().view(-1)
                    # deviation_BF_list = deviation_BF[key].permute(1, 2, 3, 0).contiguous().view(-1)
                # else:
                    # centroid_CL_list = centroid_CL[key].contiguous().view(-1)
                    # centroid_BF_list = centroid_BF[key].contiguous().view(-1)
                    # deviation_CL_list = deviation_CL[key].contiguous().view(-1)
                    # deviation_BF_list = deviation_BF[key].contiguous().view(-1)

                # print_log("Attacking which layer:", log)
                # print_log(key, log)
                # print_log("Gradient Value at Attacking layer:", log)
                # print_log(attack_gradient, log)
                # print_log("Centroid Value at Attacking Point", log)
                # print_log(centroid_CL_list[attack_index], log)
                # print_log("------------", log)
                # print_log(centroid_BF_list[attack_index], log)
                # print_log("Centroid Value prior to Attacking Point", log)
                # print_log(centroid_CL_list[attack_index-5:attack_index+5], log)
                # print_log("------------", log)
                # print_log(centroid_BF_list[attack_index-5:attack_index+5], log)
                # print_log("Deviation Value at Attacking Point", log)
                # print_log(deviation_CL_list[attack_index], log)
                # print_log("------------", log)
                # print_log(deviation_BF_list[attack_index], log)
                # print_log("Deviation Value prior to Attacking Point", log)
                # print_log(deviation_CL_list[attack_index-5:attack_index+5], log)
                # print_log("------------", log)
                # print_log(deviation_BF_list[attack_index-5:attack_index+5], log)
    return val_acc_top1, val_loss, gradient_map, val_acc_top1_no_recover

def perform_oneshotattack(attacker, model, model_check, train_loader, test_loader, log):
    # Note that, attack has to be done in evaluation model due to batch-norm.
    # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
    model.eval()
    losses = AverageMeter()
    iter_time = AverageMeter()
    attack_time = AverageMeter()
    # attempt to use the training data to conduct BFA
    for _, (data, target) in enumerate(train_loader):
        if args.use_cuda:
            target = target.cuda(async=True)
            data = data.cuda()
        # Override the target to prevent label leaking
        _, target = model(data).data.max(1)
        break

    print_log('k_top is set to {}'.format(args.k_top), log)
    print_log('Attack sample size is {}'.format(data.size()[0]), log)
    end = time.time()

    print_log('**********************************', log)
    attacker.oneshot_attack_apply(model, data, target)

    # measure data loading time
    attack_time.update(time.time() - end)
    end = time.time()

    print_log(
        'Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f})  '.
        format(attack_time=attack_time,
               iter_time=iter_time) + time_string(), log)

    # exam the BFA on entire val dataset
    val_acc_top1, val_acc_top5, val_loss= validate(test_loader, model, attacker.criterion, log)

    print_log("End of searching, dump all the bits to the target system and perform the attack", log)
    model_check.load_state_dict(model.state_dict())
    val_acc_top1, val_acc_top5, val_loss, centroid_BF, deviation_BF = validate_new(test_loader, model_check, attacker.criterion, log)
    print_log("\n", log)
    print_log('actual loss: {:.4f}'.format(val_loss),log)
    iter_time.update(time.time() - end)
    print_log(
        'iteration Time {iter_time.val:.3f} ({iter_time.avg:.3f})'.format(
            iter_time=iter_time), log)
    end = time.time()

    return val_acc_top1, val_loss

def perform_randomattack(attacker, model, model_check, train_loader, test_loader, log):
    # Note that, attack has to be done in evaluation model due to batch-norm.
    # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
    model.eval()
    losses = AverageMeter()
    iter_time = AverageMeter()
    attack_time = AverageMeter()
    # attempt to use the training data to conduct BFA
    for _, (data, target) in enumerate(train_loader):
        if args.use_cuda:
            target = target.cuda(async=True)
            data = data.cuda()
        # Override the target to prevent label leaking
        _, target = model(data).data.max(1)
        break

    print_log('k_top is set to {}'.format(args.k_top), log)
    print_log('Attack sample size is {}'.format(data.size()[0]), log)
    end = time.time()

    print_log('**********************************', log)

    attacker.random_attack_apply(model)

    # measure data loading time
    attack_time.update(time.time() - end)
    end = time.time()

    # record the loss
    # losses.update(attacker.loss_max, data.size(0))

    print_log(
        'Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f})  '.
        format(attack_time=attack_time,
               iter_time=iter_time) + time_string(), log)

    val_acc_top1, val_acc_top5, val_loss= validate(test_loader, model, attacker.criterion, log)
    print_log("End of searching, dump all the bits to the target system and perform the attack", log)
    model_check.load_state_dict(model.state_dict())
    val_acc_top1, val_acc_top5, val_loss, centroid_BF, deviation_BF = validate_new(test_loader, model_check, attacker.criterion, log)
    print_log("\n", log)

    # measure elapsed time
    iter_time.update(time.time() - end)
    print_log(
        'iteration Time {iter_time.val:.3f} ({iter_time.avg:.3f})'.format(
            iter_time=iter_time), log)
    end = time.time()


    return val_acc_top1, val_loss


def perform_weight_shifting_attack(attacker, model, model_check, train_loader, test_loader, log, option = "weight_shuffle"):
    # Note that, attack has to be done in evaluation model due to batch-norm.
    # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
    model.eval()
    losses = AverageMeter()
    iter_time = AverageMeter()
    attack_time = AverageMeter()
    # attempt to use the training data to conduct BFA
    for _, (data, target) in enumerate(train_loader):
        if args.use_cuda:
            target = target.cuda(async=True)
            data = data.cuda()
        # Override the target to prevent label leaking
        _, target = model(data).data.max(1)
        break

    print_log('k_top is set to {}'.format(args.k_top), log)
    print_log('Attack sample size is {}'.format(data.size()[0]), log)
    end = time.time()

    print_log('**********************************', log)

    attacker.random_attack_apply(model, option)

    # measure data loading time
    attack_time.update(time.time() - end)
    end = time.time()

    print_log(
        'Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f})  '.
        format(attack_time=attack_time,
               iter_time=iter_time) + time_string(), log)

    val_acc_top1, val_acc_top5, val_loss= validate(test_loader, model, attacker.criterion, log)
    print_log("End of searching, dump all the bits to the target system and perform the attack", log)
    model_check.load_state_dict(model.state_dict())
    val_acc_top1, val_acc_top5, val_loss, centroid_BF, deviation_BF = validate_new(test_loader, model_check, attacker.criterion, log)
    print_log("\n", log)

    # measure elapsed time
    iter_time.update(time.time() - end)
    print_log(
        'iteration Time {iter_time.val:.3f} ({iter_time.avg:.3f})'.format(
            iter_time=iter_time), log)
    end = time.time()

    return val_acc_top1, val_loss

def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                target = target.cuda(async=True)
                input = input.cuda()

            # compute output
            output = model(input)

            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

        print_log(
            '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {losses.avg:.4f} Error@1 {error1:.3f}'
            .format(top1=top1, top5=top5, losses=losses, error1=100 - top1.avg), log)

    return top1.avg, top5.avg, losses.avg

def quick_detection(val_loader, model, criterion, log):
    ti = 0
    centroid = {}
    for name, m in model.named_modules():
        if ("conv" in name) or ("classifier" in name):
            ti = ti + 1
            centroid[name] = get_centroid(m.weight.data, m.grain_size, m.num_bits, m.M2D, m.half_lvls)
            weight_d = (m.weight.data - centroid[name])

            signature = get_qcode(weight_d, [1,m.check_gsize], m.check_bit, m.half_lvls, m.check_factor, m.codebook)
            if not(os.path.exists(m.save_path + '/saved_tensors/signature{}.pt'.format(ti))):
                torch.save(signature, m.save_path + '/saved_tensors/signature{}.pt'.format(ti))
            else:
                orig_signature = torch.load(m.save_path + '/saved_tensors/signature{}.pt'.format(ti))
                inequal = (signature - orig_signature != 0)
                detected_item = torch.nonzero(inequal[::2] + inequal[1::2]).size(0)
                if detected_item != 0:
                    config.check += detected_item
    return 0
# def quick_detection(val_loader, model, criterion, log):
    # model.eval()
    # with torch.no_grad():
        # input, _ = next(iter(val_loader))
        # input_image = input[0:2]
        # if args.use_cuda:
            # input_image = input_image.cuda()
            # output = model(input_image)
        # return 0
def validate_new(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                target = target.cuda(async=True)
                input = input.cuda()

            # compute output
            output = model(input)
            centroid = {}
            deviation = {}
            for name, m in model.named_modules():
                if ("conv" in name) or ("classifier" in name):
                    centroid[name] = get_centroid(m.weight.data, m.grain_size, m.num_bits, m.M2D, m.half_lvls)
                    weight_d = (m.weight.data - centroid[name])
                    clipper = (1-m.M2D) * m.half_lvls
                    deviation[name] = weight_d.clamp_(-clipper, clipper).round_()
            # print(centroid)
            # deviation_item = torch.load(save_path + '/saved_tensors/deviation{}.pt'.format(ti))
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

        print_log(
            '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {losses.avg:.4f} Error@1 {error1:.3f}'
            .format(top1=top1, top5=top5, losses=losses, error1=100 - top1.avg), log)

    return top1.avg, top5.avg, losses.avg, centroid, deviation

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_logger(base_dir, epoch, train_accuracy, test_accuracy):
    file_name = 'accuracy.txt'
    file_path = "%s/%s" % (base_dir, file_name)
    # create and format the log file if it does not exists
    if not os.path.exists(file_path):
        create_log = open(file_path, 'w')
        create_log.write('epochs train test\n')
        create_log.close()

    recorder = {}
    recorder['epoch'] = epoch
    recorder['train'] = train_accuracy
    recorder['test'] = test_accuracy
    # append the epoch index, train accuracy and test accuracy:
    with open(file_path, 'a') as accuracy_log:
        accuracy_log.write(
            '{epoch}       {train}    {test}\n'.format(**recorder))


if __name__ == '__main__':
    main()
