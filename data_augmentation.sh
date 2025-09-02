#!/bin/bash

#SBATCH --account=jorneirik 
#SBATCH --job-name=cifar10_adversarial_training
#SBATCH --output=cifar10_adversarial_training.log   
#SBATCH --partition=hgx2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 6          #  På g001 bør en berense seg til 96 / 16 cores per GPU
#SBATCH --gres=gpu:1

srun -n 1 python training_scheme_data_augmentation.py dataset=cifar10 model=resnet18 batch_norm=True lr=0.01 eval_interval=5 epochs=301
srun -n 1 python training_scheme_data_augmentation.py dataset=cifar10 model=resnet18 batch_norm=True lr=0.01 eval_interval=5 dropout=True epochs=301
