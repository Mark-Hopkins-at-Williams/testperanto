#!/bin/sh

#SBATCH -c 32 # Request 32 CPU cores
#SBATCH -t 0-02:00 # Runtime in D-HH:MM
#SBATCH -p dl # Partition to submit to
#SBATCH --mem=2G # Request 2G of memory
#SBATCH -o /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/output/output.out # File to which STDOUT will be written
#SBATCH -e /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/output/error.err # File to which STDERR will be written
#SBATCH --gres=gpu:0 # Request 0 GPUs
parallel --jobs 32 <<EOT
python /mnt/storage/jcheigh/testperanto/scripts/parallel_gen.py -c /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/svo_permutations.yaml -n 100 -o /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/output/svo_permutations100
python /mnt/storage/jcheigh/testperanto/scripts/parallel_gen.py -c /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/svo_permutations.yaml -n 200 -o /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/output/svo_permutations200
python /mnt/storage/jcheigh/testperanto/scripts/parallel_gen.py -c /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/svo_permutations.yaml -n 300 -o /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/output/svo_permutations300
EOT
