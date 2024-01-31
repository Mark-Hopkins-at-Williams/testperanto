#!/bin/bash
#SBATCH -c 20               # Number of cores (-c)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p dl               # Partition to submit to
#SBATCH -o log_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e log_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --gres=gpu:1        # Request GPUs

SRC=$3
TGT=$4
MAX_EPOCHS=$5

mkdir $1
cp -R $2 $1/

###########
# THIS PART OF THE CODE FIGURES OUT THE DIRECTORY OF THE SHELL SCRIPT.
# THIS IS SOMEWHAT COMPLICATED WHEN WE RUN THROUGH SLURM.
# check if script is started via SLURM or bash
# if with SLURM: there variable '$SLURM_JOB_ID' will exist
# `if [ -n $SLURM_JOB_ID ]` checks if $SLURM_JOB_ID is not an empty string
if [ -n $SLURM_JOB_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_COMMAND=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_COMMAND=$(realpath $0)
fi
SCRIPT_COMMAND_ARRAY=($SCRIPT_COMMAND)
SCRIPT_NAME=${SCRIPT_COMMAND_ARRAY[0]}
SCRIPT_PATH=$(dirname $SCRIPT_NAME)
#
###########

bash $SCRIPT_PATH/prepare-data.sh $1 $SRC $TGT

TEXT=$1/data-tokenized
BINARY_TEXT=$1/data-bin

fairseq-preprocess --source-lang $SRC --target-lang $TGT \
    --trainpref $TEXT/train --validpref $TEXT/dev --testpref $TEXT/test \
    --destdir $BINARY_TEXT \
    --scoring chrf \
    --workers 20

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    $BINARY_TEXT \
    --max-epoch $MAX_EPOCHS \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --scoring chrf \
    --no-epoch-checkpoints \
    --save-dir $1/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

fairseq-generate $BINARY_TEXT \
    --path $1/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > $1/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py $1/translations

sacrebleu $1/translations.ref -i $1/translations.hyp -m bleu chrf > $1/scores