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
export CONFIG=$PRJ/brainage/config
export CUDA_VISIBLE_DEVICES=0
cd $PRJ/brainage
#python $PRJ/brainage/experiments/gradcam_patch.py /home/raheppt1/projects/brainage/outputs/brainage/patchwise/large/fold4/2020-09-11/03-50-25/brainage/2x0hdrxb/checkpoints/epoch=198.ckpt
python $PRJ/brainage/experiments/gradcam_patch.py /home/raheppt1/projects/brainage/outputs/brainage/patchwise/large/fold3/2020-09-08/22-03-45/brainage/104b8ebn/checkpoints/epoch=178.ckpt
python $PRJ/brainage/experiments/gradcam_patch.py /home/raheppt1/projects/brainage/outputs/brainage/patchwise/large/fold2/2020-08-30/20-18-46/brainage/vyl8rb7d/checkpoints/epoch=209.ckpt
python $PRJ/brainage/experiments/gradcam_patch.py /home/raheppt1/projects/brainage/outputs/brainage/patchwise/large/fold1/2020-08-30/20-18-46/brainage/3iuh1vgc/checkpoints/epoch=233.ckpt
python $PRJ/brainage/experiments/gradcam_patch.py /home/raheppt1/projects/brainage/outputs/brainage/patchwise/large/fold0/2020-08-30/20-18-46/brainage/301f9tu6/checkpoints/epoch=243.ckpt