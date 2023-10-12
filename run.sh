#!/bin/sh
#SBATCH -c 1                # Request 1 CPU core
#SBATCH -t 0-02:00          # Runtime in D-HH:MM, minimum of 10 mins
#SBATCH -p dl               # Partition to submit to 
#SBATCH --mem=10G           # Request 10G of memory
#SBATCH -o osv_output_%j.out  # File to which STDOUT will be written
#SBATCH -e osv_errors_%j.err  # File to which STDERR will be written
#SBATCH --gres=gpu:2        # Request two GPUs
python scripts/generate.py -c examples/svo/amr.json examples/svo/middleman.json examples/permutations/englishOSV.json --sents -n 5

