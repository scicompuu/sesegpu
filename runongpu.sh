#!/bin/sh
#SBATCH -A g2021027 -t 0:59:00 -p core -n 4 -M snowy --gres=gpu:t4:1 --parsable
singularity run --nv /proj/g2020014/nobackup/private/$@
