#!/bin/sh

#SBATCH -c 32 # Request 32 CPU cores
#SBATCH -t 0-10:00 # Runtime in D-HH:MM
#SBATCH -p dl # Partition to submit to
#SBATCH --mem=2G # Request 2G of memory
#SBATCH -o /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/output/outputA.out # File to which STDOUT will be written
#SBATCH -e /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/output/errorA.err # File to which STDERR will be written
#SBATCH --gres=gpu:1 # Request 1 GPU
parallel --jobs 32 <<EOT
python /mnt/storage/jcheigh/testperanto/scripts/parallel_gen.py -c /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/svo_permutations.yaml -n 1000 -o /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/output/svo_permutationsA1.0k
python /mnt/storage/jcheigh/testperanto/scripts/parallel_gen.py -c /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/svo_permutations.yaml -n 2000 -o /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/output/svo_permutationsA2.0k
python /mnt/storage/jcheigh/testperanto/scripts/parallel_gen.py -c /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/svo_permutations.yaml -n 4000 -o /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/output/svo_permutationsA4.0k
python /mnt/storage/jcheigh/testperanto/scripts/parallel_gen.py -c /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/svo_permutations.yaml -n 8000 -o /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/output/svo_permutationsA8.0k
python /mnt/storage/jcheigh/testperanto/scripts/parallel_gen.py -c /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/svo_permutations.yaml -n 16000 -o /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/output/svo_permutationsA16.0k
python /mnt/storage/jcheigh/testperanto/scripts/parallel_gen.py -c /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/svo_permutations.yaml -n 32000 -o /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/output/svo_permutationsA32.0k
python /mnt/storage/jcheigh/testperanto/scripts/parallel_gen.py -c /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/svo_permutations.yaml -n 64000 -o /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/output/svo_permutationsA64.0k
python /mnt/storage/jcheigh/testperanto/scripts/parallel_gen.py -c /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/svo_permutations.yaml -n 128000 -o /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/output/svo_permutationsA128.0k
python /mnt/storage/jcheigh/testperanto/scripts/parallel_gen.py -c /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/svo_permutations.yaml -n 256000 -o /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/output/svo_permutationsA256.0k
EOT
