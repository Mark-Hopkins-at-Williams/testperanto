#!/bin/sh
#SBATCH -c 1
#SBATCH -t 0-05:00
#SBATCH -p dl
#SBATCH --mem=10G
#SBATCH -o myoutput_%j.out
#SBATCH -e myerrors_%j.err
#SBATCH --gres=gpu:1
python3 experiment.py 