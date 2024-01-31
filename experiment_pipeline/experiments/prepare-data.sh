#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

BPE_TOKENS=2000
SCRIPT_PATH=$4
BPEROOT=$SCRIPT_PATH/subword-nmt/subword_nmt

src=$2
tgt=$3
lang=$2-$3
orig=$1/data
prep=$1/data-tokenized

mkdir -p $prep

TRAIN=$orig/train.$src-$tgt
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $orig/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L dev.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $orig/$f > $prep/$f
    done
done
