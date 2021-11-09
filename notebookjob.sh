#!/bin/sh
#SBATCH -A g2021027 -t 0:59:00 -p core -n 4 -M snowy --gres=gpu:t4:1 --parsable
CENV=${5-tf-gpu}
CONTAINER=${6-container.sif}
singularity run --nv /proj/g2020014/nobackup/private/$CONTAINER -c "conda activate $CENV && jupyter lab --ip=127.0.0.1 --port=$1" && exit
singularity run --nv /proj/g2020014/nobackup/private/$CONTAINER -c "conda activate $CENV && jupyter lab --ip=127.0.0.1 --port=$2" && exit
singularity run --nv /proj/g2020014/nobackup/private/$CONTAINER -c "conda activate $CENV && jupyter lab --ip=127.0.0.1 --port=$3" && exit
singularity run --nv /proj/g2020014/nobackup/private/$CONTAINER -c "conda activate $CENV && jupyter lab --ip=127.0.0.1 --port=$4" && exit
