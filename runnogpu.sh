#!/bin/sh
#SBATCH -A g2021027 -t 0:59:00 -p devcore -n 4 -M snowy --parsable
singularity run /proj/g2020014/nobackup/private/$@
