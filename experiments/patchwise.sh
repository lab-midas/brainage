#!/bin/bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
export PRJ=/home/raheppt1/projects
export DATA=/mnt/qdata/raheppt1/data/brainage
export OUT=/home/raheppt1/projects/brainage/outputs

pyenv deactivate
pyenv activate brainage2
echo "$(pyenv which python)"
export CONFIG=$PRJ/brainage/config/patchwise/config.yaml
export CUDA_VISIBLE_DEVICES=1
cd $PRJ/brainage
python $PRJ/brainage/brainage/trainer/train3d.py dataset=hr dataset.fold=0
python $PRJ/brainage/brainage/trainer/train3d.py dataset=hr dataset.fold=1
python $PRJ/brainage/brainage/trainer/train3d.py dataset=hr dataset.fold=2
python $PRJ/brainage/brainage/trainer/train3d.py dataset=hr dataset.fold=3
python $PRJ/brainage/brainage/trainer/train3d.py dataset=hr dataset.fold=4