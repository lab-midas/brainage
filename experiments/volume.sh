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
export CONFIG=$PRJ/brainage/config/volume/config.yaml
export CUDA_VISIBLE_DEVICES=0
cd $PRJ/brainage
python $PRJ/brainage/brainage/trainer/train3d.py dataset.fold=0,1,2,3,4 -m