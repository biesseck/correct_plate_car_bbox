#!/bin/bash

ENV_NAME=correct_plate_car_bbox

source ~/anaconda3/etc/profile.d/conda.sh

conda create --name $ENV_NAME python=3.9 --yes
conda activate $ENV_NAME

pip3 install -r requirements.txt
