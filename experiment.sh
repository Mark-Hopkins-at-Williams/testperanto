#!/bin/bash
CONFIG_FILE=$1
NUM_SENTS=1000
SWITCHES=$2
OUTPUT_DIR=$3
echo "Generating data."
python scripts/generate_dataset.py \
  -c $CONFIG_FILE \
  -n $NUM_SENTS \
  --sents \
  -s $SWITCHES \
  -d $OUTPUT_DIR
echo "Preprocessing data."
fairseq-preprocess \
  --only-source \
  --trainpref $OUTPUT_DIR/sents.train \
  --validpref $OUTPUT_DIR/sents.valid \
  --testpref $OUTPUT_DIR/sents.test \
  --destdir $OUTPUT_DIR/fairseq
echo "Training."
fairseq-train \
  $OUTPUT_DIR/fairseq \
  --task language_modeling \
  --tensorboard-logdir $OUTPUT_DIR/tensorboard \
  --save-dir $OUTPUT_DIR/checkpoints \
  --arch transformer_lm \
  --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --weight-decay 0.01 \
  --clip-norm 0.0 \
  --lr 0.0005 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 4000 \
  --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 \
  --sample-break-mode none \
  --max-tokens 2048 \
  --update-freq 16 \
  --fp16 \
  --max-update 50000

