#!/bin/bash

#SBATCH -c 8 # Request 8 CPU cores
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -p dl # Partition to submit to
#SBATCH --mem=2G # Request 2G of memory
#SBATCH -o /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/output0.out # File to which STDOUT will be written
#SBATCH -e /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/error0.err # File to which STDERR will be written
#SBATCH --gres=gpu:1 # Request 1 GPUs

SRC=en
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_1.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_SVO_1.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_1.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_1.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_1.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_1.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_1.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_1.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_1.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_1.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_1.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_1.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_1.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_1.0k/scores


SRC=fr
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_1.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_VSO_1.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_1.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_1.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_1.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_1.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_1.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_1.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_1.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_1.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_1.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_1.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_1.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_1.0k/scores


SRC=ko
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_1.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OSV_OSV_1.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_1.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_1.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_1.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_1.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_1.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_1.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_1.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_1.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_1.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_1.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_1.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_1.0k/scores


SRC=en
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_2.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_SVO_2.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_2.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_2.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_2.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_2.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_2.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_2.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_2.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_2.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_2.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_2.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_2.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_2.0k/scores


SRC=fr
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_2.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_VSO_2.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_2.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_2.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_2.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_2.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_2.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_2.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_2.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_2.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_2.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_2.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_2.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_2.0k/scores


SRC=ko
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_2.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OSV_OSV_2.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_2.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_2.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_2.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_2.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_2.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_2.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_2.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_2.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_2.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_2.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_2.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_2.0k/scores


SRC=en
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_4.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_SVO_4.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_4.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_4.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_4.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_4.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_4.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_4.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_4.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_4.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_4.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_4.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_4.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_4.0k/scores


SRC=fr
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_4.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_VSO_4.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_4.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_4.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_4.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_4.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_4.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_4.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_4.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_4.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_4.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_4.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_4.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_4.0k/scores


SRC=ko
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_4.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OSV_OSV_4.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_4.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_4.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_4.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_4.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_4.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_4.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_4.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_4.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_4.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_4.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_4.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_4.0k/scores


SRC=en
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_8.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_SVO_8.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_8.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_8.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_8.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_8.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_8.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_8.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_8.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_8.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_8.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_8.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_8.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_8.0k/scores


SRC=fr
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_8.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_VSO_8.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_8.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_8.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_8.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_8.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_8.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_8.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_8.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_8.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_8.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_8.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_8.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_8.0k/scores


SRC=ko
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_8.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OSV_OSV_8.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_8.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_8.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_8.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_8.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_8.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_8.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_8.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_8.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_8.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_8.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_8.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_8.0k/scores


SRC=en
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_16.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_SVO_16.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_16.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_16.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_16.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_16.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_16.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_16.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_16.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_16.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_16.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_16.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_16.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_16.0k/scores


SRC=fr
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_16.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_VSO_16.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_16.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_16.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_16.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_16.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_16.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_16.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_16.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_16.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_16.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_16.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_16.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_16.0k/scores


SRC=ko
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_16.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OSV_OSV_16.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_16.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_16.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_16.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_16.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_16.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_16.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_16.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_16.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_16.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_16.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_16.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_16.0k/scores


SRC=en
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_32.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_SVO_32.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_32.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_32.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_32.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_32.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_32.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_32.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_32.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_32.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_32.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_32.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_32.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_32.0k/scores


SRC=fr
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_32.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_VSO_32.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_32.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_32.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_32.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_32.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_32.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_32.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_32.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_32.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_32.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_32.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_32.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_32.0k/scores


SRC=ko
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_32.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OSV_OSV_32.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_32.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_32.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_32.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_32.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_32.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_32.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_32.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_32.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_32.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_32.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_32.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_32.0k/scores


SRC=de
TGT=it
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_64.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_OVS_64.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_64.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_64.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_64.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_64.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_64.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_64.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_64.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_64.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_64.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_64.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_64.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_64.0k/scores


SRC=fr
TGT=ko
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_64.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_OSV_64.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_64.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_64.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_64.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_64.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_64.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_64.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_64.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_64.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_64.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_64.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_64.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_64.0k/scores


SRC=es
TGT=ko
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_64.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_OSV_64.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_64.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_64.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_64.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_64.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_64.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_64.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_64.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_64.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_64.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_64.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_64.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_64.0k/scores


SRC=ko
TGT=it
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_64.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OSV_OVS_64.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_64.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_64.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_64.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_64.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_64.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_64.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_64.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_64.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_64.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_64.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_64.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_64.0k/scores


SRC=de
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_64.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_SOV_64.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_64.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_64.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_64.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_64.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_64.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_64.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_64.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_64.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_64.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_64.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_64.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_64.0k/scores


SRC=es
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_64.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_VOS_64.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_64.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_64.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_64.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_64.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_64.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_64.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_64.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_64.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_64.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_64.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_64.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_64.0k/scores


SRC=it
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_64.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OVS_OVS_64.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_64.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_64.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_64.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_64.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_64.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_64.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_64.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_64.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_64.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_64.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_64.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_64.0k/scores


SRC=en
TGT=fr
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_VSO_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_128.0k/scores


SRC=en
TGT=ko
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_OSV_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_128.0k/scores


SRC=de
TGT=fr
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_VSO_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_128.0k/scores


SRC=de
TGT=ko
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_OSV_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_128.0k/scores


SRC=fr
TGT=es
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_VOS_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_128.0k/scores


SRC=fr
TGT=it
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_OVS_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_128.0k/scores


SRC=es
TGT=it
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_OVS_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_128.0k/scores


SRC=en
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_SVO_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO_128.0k/scores


SRC=fr
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_VSO_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO_128.0k/scores


SRC=ko
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OSV_OSV_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV_128.0k/scores


SRC=en
TGT=de
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_SOV_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_256.0k/scores


SRC=en
TGT=es
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_VOS_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_256.0k/scores


SRC=en
TGT=it
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_OVS_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_256.0k/scores


SRC=de
TGT=es
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_VOS_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_256.0k/scores


SRC=de
TGT=it
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_OVS_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_256.0k/scores


SRC=fr
TGT=ko
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_OSV_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_256.0k/scores


SRC=es
TGT=ko
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_OSV_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_256.0k/scores


SRC=ko
TGT=it
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OSV_OVS_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_256.0k/scores


SRC=de
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_SOV_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV_256.0k/scores


SRC=es
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_VOS_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS_256.0k/scores


SRC=it
TGT=da
MAX_EPOCHS=1000

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OVS_OVS_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 40 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS_256.0k/scores

