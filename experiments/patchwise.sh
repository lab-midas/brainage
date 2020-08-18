#!/bin/bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
export PRJ=/home/raheppt1/projects
export DATA=/mnt/qdata/raheppt1/data/brainage
export OUT=/home/raheppt1/projects/brainage/outputs

pyenv deactivate
pyenv activate brainage
echo "$(pyenv which python)"
export CONFIG=$PRJ/brainage/config/patchwise/config.yaml
export CUDA_VISIBLE_DEVICES=1
cd $PRJ/brainage
python $PRJ/brainage/brainage/trainer/train3d.py dataset=hr dataset.fold=0 dataset.patch_size=[64,64,64]  
python $PRJ/brainage/brainage/trainer/train3d.py dataset=hr dataset.fold=0 dataset.patch_size=[48,48,48]  