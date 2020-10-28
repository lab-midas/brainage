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
python $PRJ/brainage/experiments/gradcam_volume.py /home/raheppt1/projects/brainage/outputs/brainage/volume/fold4/2020-09-08/22-38-24/brainage/382jap2s/checkpoints/epoch=161.ckpt
python $PRJ/brainage/experiments/gradcam_volume.py /home/raheppt1/projects/brainage/outputs/brainage/volume/fold3/2020-09-07/10-26-26/brainage/27lr0xi8/checkpoints/epoch=178.ckpt
python $PRJ/brainage/experiments/gradcam_volume.py /home/raheppt1/projects/brainage/outputs/brainage/volume/fold2/2020-09-03/13-57-47/brainage/1clvds22/checkpoints/epoch=188.ckpt
python $PRJ/brainage/experiments/gradcam_volume.py /home/raheppt1/projects/brainage/outputs/brainage/volume/fold1/2020-09-01/02-59-55/brainage/37myxdpl/checkpoints/epoch=186.ckpt
python $PRJ/brainage/experiments/gradcam_volume.py /home/raheppt1/projects/brainage/outputs/brainage/volume/fold0/2020-08-30/13-54-03/brainage/3kbt2k6i/checkpoints/epoch=243.ckpt