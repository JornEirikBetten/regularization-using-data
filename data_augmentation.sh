#!/bin/bash

#SBATCH --account=jorneirik 
#SBATCH --job-name=cifar10_adversarial_training
#SBATCH --output=cifar10_adversarial_training.log   
#SBATCH --partition=dgx2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 6          #  På g001 bør en berense seg til 96 / 16 cores per GPU
#SBATCH --gres=gpu:1


srun -n 1 python training_scheme_data_augmentation.py dataset=cifar10 model=resnet18 batch_norm=True lr=0.02 eval_interval=10 dropout=True batch_size=512 total_updates=50000
srun -n 1 python training_scheme_data_augmentation.py dataset=cifar10 model=resnet18 batch_norm=True lr=0.02 eval_interval=10 dropout=True batch_size=256 total_updates=50000
srun -n 1 python training_scheme_data_augmentation.py dataset=cifar10 model=resnet18 batch_norm=True lr=0.02 eval_interval=10 dropout=True batch_size=128 total_updates=50000
srun -n 1 python training_scheme_data_augmentation.py dataset=cifar10 model=resnet18 batch_norm=True lr=0.02 eval_interval=10 dropout=True batch_size=64 total_updates=50000
# srun -n 1 python training_scheme_data_augmentation.py dataset=cifar10 model=resnet18 batch_norm=True lr=0.02 eval_interval=10 dropout=True batch_size=32
# srun -n 1 python training_scheme_data_augmentation.py dataset=cifar10 model=resnet18 batch_norm=True lr=0.02 eval_interval=10 dropout=True batch_size=16
