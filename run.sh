#!/bin/bash

#SBATCH --account=jorneirik 
#SBATCH --job-name=adversarial_training
#SBATCH --output=adversarial_training.log   
#SBATCH --partition=dgx2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 6          #  På g001 bør en berense seg til 96 / 16 cores per GPU
#SBATCH --gres=gpu:1

srun -n 1 python training_scheme_data_augmentation.py dataset=cifar10 model=resnet18 batch_norm=True dropout=True eval_interval=10 epochs=101 lr=0.1 batch_size=128
srun -n 1 python training_scheme_data_augmentation.py dataset=cifar10 model=resnet18 batch_norm=True dropout=True eval_interval=10 epochs=101 lr=0.01 batch_size=128
srun -n 1 python training_scheme_data_augmentation.py dataset=cifar10 model=resnet18 batch_norm=True dropout=True eval_interval=10 epochs=101 lr=0.001 batch_size=128 