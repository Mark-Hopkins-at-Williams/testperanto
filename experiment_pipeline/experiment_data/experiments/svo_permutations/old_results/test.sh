#!/bin/sh

#SBATCH -c 8 # Request 8 CPU cores
#SBATCH -t 2-00:00 # Runtime in D-HH:MM
#SBATCH -p dl # Partition to submit to
#SBATCH --mem=2G # Request 2G of memory
#SBATCH -o /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/output.out # File to which STDOUT will be written
#SBATCH -e /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/error.err # File to which STDERR will be written
#SBATCH --gres=gpu:2 # Request 2 GPUs
OUTPUT=$( sbatch /mnt/storage/hopkins/mt/appa-mt/fairseq/train.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_1.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_SOV_1.0k en de 5 )
JOB_ID=$(echo $OUTPUT | awk '{print $NF}')
OUTPUT=$( sbatch /mnt/storage/hopkins/mt/appa-mt/fairseq/train.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_1.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_VSO_1.0k en fr 5 )
JOB_ID=$(echo $OUTPUT | awk '{print $NF}')
srun -n 1 -c 1 --mem=1M sh -c "while squeue -j $JOB_ID | grep -q ' R\| PD'; do sleep 60; done"
OUTPUT=$( sbatch /mnt/storage/hopkins/mt/appa-mt/fairseq/train.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_1.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_VOS_1.0k en es 5 )
JOB_ID=$(echo $OUTPUT | awk '{print $NF}')
srun -n 1 -c 1 --mem=1M sh -c "while squeue -j $JOB_ID | grep -q ' R\| PD'; do sleep 60; done"
OUTPUT=$( sbatch /mnt/storage/hopkins/mt/appa-mt/fairseq/train.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_1.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_OSV_1.0k en ko 5 )
JOB_ID=$(echo $OUTPUT | awk '{print $NF}')
srun -n 1 -c 1 --mem=1M sh -c "while squeue -j $JOB_ID | grep -q ' R\| PD'; do sleep 60; done"
EOT