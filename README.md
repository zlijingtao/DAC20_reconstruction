  
  
  
# Defending Bif-Flip Attack using Weight Reconstruction
  
  
This repository contains a Pytorch implementation of the paper "[Defending Bit-Flip Attack through DNN Weight Reconstruction](https://ieeexplore.ieee.org/abstract/document/9218665 )".
  
If you find this project useful to you, please cite [our work](https://ieeexplore.ieee.org/abstract/document/9218665 ):
  
  
```
@inproceedings{li2020defending,
  title={Defending bit-flip attack through DNN weight reconstruction},
  author={Li, Jingtao and Rakin, Adnan Siraj and Xiong, Yan and Chang, Liangliang and He, Zhezhi and Fan, Deliang and Chakrabarti, Chaitali},
  booktitle={2020 57th ACM/IEEE Design Automation Conference (DAC)},
  pages={1--6},
  year={2020},
  organization={IEEE}
}
```
  
## Dependencies:
  
  
* Python 3.6 (Anaconda)
* Create Environment from ./environment.yml
  
  
## Usage 
  
  
For getting Fig.7 in the paper, which is to test the resistance to BFA on pretrained model(with/without weight reconstruction), please use the following command in terminal.


CIFAR-10:
```bash
bash BFA_attack_test_CIFAR10.sh
```
  
ImageNet:
```bash
bash BFA_attack_test_ImageNet.sh (not yet available, please file an issue if your need this)
```

For getting how many number will crush the model (degrade to 10.00% accuracy. a.k.a. BFA stress test):

CIFAR-10:
```bash
bash BFA_stress_test_CIFAR10.sh
```

############ directory to save result #############
DATE=`date +%Y-%m-%d`
  
if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./save/${DATE}/
fi
  
## Task list
  
- [x] Upload Trained models for CIFAR-10 datasets.
  
- [] Upload Trained models for ImageNet datasets.
  
  
  