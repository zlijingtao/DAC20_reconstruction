#!/usr/bin/env sh
#Set the python to your conda environment
PYTHON="/opt/anaconda3/envs/pytorch_p36/bin/python"
############ directory to save result #############
DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
    mkdir ./save/${DATE}/
fi

############ Configurations ###############

dataset=cifar10
batch_size=128
label_info=test
model=quan_resnet20
max_layer=20
gpuid=0
############ BFA Configurations ###############

test_batch_size=256
attack_sample_size=128 # number of data used for PBFA
n_iter=20 # number of iteration to perform PBFA (actuall total number of bits flipped in PBFA)
k_top=40 # Number of bits that is taken into candidates (PBFA)/ flipped per layer(RBFA/Oneshot): PBFA set to 40, RBFA set to 20, Oneshot set to 1.
layer_id=0 # 1 - 20, indicate which layer to attack, set to 0 indicates attacking every layer with the same rate
massive=5


ingrain_d2=4
inbit=2
inM2D=0.8
resgrain_d2=4
resbit=2
resM2D=0.8
outgrain_d2=4
outbit=2
outM2D=0.8

resume_path="./test/cifar10_quan_resnet20_200_i4r4o4_adam0.8_2bit/model_best.pth.tar"

python test_main.py --dataset ${dataset} \
    --data_path ./dataset/   \
    --arch ${model} --resume ${resume_path} --massive ${massive}\
    --input_grain_size 1 ${ingrain_d2} --input_num_bits ${inbit} --input_M2D ${inM2D} \
    --res_grain_size 1 ${resgrain_d2} --res_num_bits ${resbit} --res_M2D ${resM2D} \
    --output_grain_size 1 ${outgrain_d2} --output_num_bits ${outbit} --output_M2D ${outM2D} \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id ${gpuid} \
    --reset_weight \
    --layer_id ${layer_id} \
    --n_iter ${n_iter} --k_top ${k_top} \
    --attack_sample_size ${attack_sample_size} --bfa

ingrain_d2=1
inbit=1
inM2D=0.0
resgrain_d2=1
resbit=1
resM2D=0.0
outgrain_d2=1
outbit=1
outM2D=0.0

resume_path="./test/cifar10_quan_resnet20_200_i1r1o1_adam0.0_1bit_no_regularize/model_best.pth.tar"

python test_main.py --dataset ${dataset} \
    --data_path ./dataset/   \
    --arch ${model} --resume ${resume_path} --massive ${massive}\
    --input_grain_size 1 ${ingrain_d2} --input_num_bits ${inbit} --input_M2D ${inM2D} \
    --res_grain_size 1 ${resgrain_d2} --res_num_bits ${resbit} --res_M2D ${resM2D} \
    --output_grain_size 1 ${outgrain_d2} --output_num_bits ${outbit} --output_M2D ${outM2D} \
    --batch_size ${batch_size} --workers 4 --ngpu 1 --gpu_id ${gpuid} \
    --reset_weight \
    --layer_id ${layer_id} \
    --n_iter ${n_iter} --k_top ${k_top} \
    --attack_sample_size ${attack_sample_size} --bfa
