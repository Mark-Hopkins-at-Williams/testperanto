#!/bin/bash

#SBATCH -c 8 # Request 8 CPU cores
#SBATCH -t 5-00:00 # Runtime in D-HH:MM
#SBATCH -p dl # Partition to submit to
#SBATCH --mem=2G # Request 2G of memory
#SBATCH -o /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/output1.out # File to which STDOUT will be written
#SBATCH -e /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/error1.err # File to which STDERR will be written
#SBATCH --gres=gpu:1 # Request 1 GPUs

SRC=svo
TGT=vso
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_1.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_VSO_1.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_1.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_1.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_1.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_1.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_1.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_1.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_1.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_1.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_1.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_1.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_1.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_1.0k/scores


SRC=svo
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_1.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_OSV_1.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_1.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_1.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_1.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_1.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_1.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_1.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_1.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_1.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_1.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_1.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_1.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_1.0k/scores


SRC=sov
TGT=vso
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_1.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_VSO_1.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_1.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_1.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_1.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_1.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_1.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_1.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_1.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_1.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_1.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_1.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_1.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_1.0k/scores


SRC=sov
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_1.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_OSV_1.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_1.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_1.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_1.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_1.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_1.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_1.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_1.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_1.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_1.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_1.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_1.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_1.0k/scores


SRC=vso
TGT=vos
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_1.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_VOS_1.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_1.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_1.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_1.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_1.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_1.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_1.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_1.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_1.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_1.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_1.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_1.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_1.0k/scores


SRC=vso
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_1.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_OVS_1.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_1.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_1.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_1.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_1.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_1.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_1.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_1.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_1.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_1.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_1.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_1.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_1.0k/scores


SRC=vos
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_1.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_OVS_1.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_1.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_1.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_1.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_1.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_1.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_1.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_1.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_1.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_1.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_1.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_1.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_1.0k/scores


SRC=svo
TGT=svo1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_1.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_SVO1_1.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_1.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_1.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_1.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_1.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_1.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_1.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_1.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_1.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_1.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_1.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_1.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_1.0k/scores


SRC=vso
TGT=vso1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_1.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_VSO1_1.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_1.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_1.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_1.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_1.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_1.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_1.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_1.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_1.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_1.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_1.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_1.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_1.0k/scores


SRC=osv
TGT=osv1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_1.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OSV_OSV1_1.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_1.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_1.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_1.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_1.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_1.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_1.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_1.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_1.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_1.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_1.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_1.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_1.0k/scores


SRC=svo
TGT=sov
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_2.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_SOV_2.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_2.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_2.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_2.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_2.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_2.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_2.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_2.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_2.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_2.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_2.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_2.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_2.0k/scores


SRC=svo
TGT=vos
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_2.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_VOS_2.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_2.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_2.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_2.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_2.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_2.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_2.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_2.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_2.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_2.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_2.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_2.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_2.0k/scores


SRC=svo
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_2.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_OVS_2.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_2.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_2.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_2.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_2.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_2.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_2.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_2.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_2.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_2.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_2.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_2.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_2.0k/scores


SRC=sov
TGT=vos
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_2.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_VOS_2.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_2.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_2.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_2.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_2.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_2.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_2.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_2.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_2.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_2.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_2.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_2.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_2.0k/scores


SRC=sov
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_2.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_OVS_2.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_2.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_2.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_2.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_2.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_2.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_2.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_2.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_2.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_2.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_2.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_2.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_2.0k/scores


SRC=vso
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_2.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_OSV_2.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_2.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_2.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_2.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_2.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_2.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_2.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_2.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_2.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_2.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_2.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_2.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_2.0k/scores


SRC=vos
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_2.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_OSV_2.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_2.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_2.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_2.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_2.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_2.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_2.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_2.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_2.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_2.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_2.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_2.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_2.0k/scores


SRC=osv
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_2.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OSV_OVS_2.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_2.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_2.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_2.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_2.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_2.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_2.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_2.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_2.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_2.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_2.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_2.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_2.0k/scores


SRC=sov
TGT=sov1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_2.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_SOV1_2.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_2.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_2.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_2.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_2.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_2.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_2.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_2.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_2.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_2.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_2.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_2.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_2.0k/scores


SRC=vos
TGT=vos1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_2.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_VOS1_2.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_2.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_2.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_2.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_2.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_2.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_2.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_2.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_2.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_2.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_2.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_2.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_2.0k/scores


SRC=ovs
TGT=ovs1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_2.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OVS_OVS1_2.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_2.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_2.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_2.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_2.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_2.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_2.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_2.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_2.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_2.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_2.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_2.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_2.0k/scores


SRC=svo
TGT=vso
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_4.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_VSO_4.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_4.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_4.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_4.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_4.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_4.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_4.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_4.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_4.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_4.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_4.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_4.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_4.0k/scores


SRC=svo
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_4.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_OSV_4.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_4.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_4.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_4.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_4.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_4.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_4.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_4.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_4.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_4.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_4.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_4.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_4.0k/scores


SRC=sov
TGT=vso
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_4.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_VSO_4.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_4.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_4.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_4.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_4.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_4.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_4.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_4.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_4.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_4.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_4.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_4.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_4.0k/scores


SRC=sov
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_4.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_OSV_4.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_4.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_4.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_4.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_4.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_4.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_4.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_4.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_4.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_4.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_4.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_4.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_4.0k/scores


SRC=vso
TGT=vos
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_4.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_VOS_4.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_4.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_4.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_4.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_4.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_4.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_4.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_4.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_4.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_4.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_4.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_4.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_4.0k/scores


SRC=vso
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_4.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_OVS_4.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_4.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_4.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_4.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_4.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_4.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_4.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_4.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_4.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_4.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_4.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_4.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_4.0k/scores


SRC=vos
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_4.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_OVS_4.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_4.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_4.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_4.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_4.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_4.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_4.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_4.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_4.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_4.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_4.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_4.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_4.0k/scores


SRC=svo
TGT=svo1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_4.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_SVO1_4.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_4.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_4.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_4.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_4.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_4.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_4.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_4.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_4.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_4.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_4.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_4.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_4.0k/scores


SRC=vso
TGT=vso1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_4.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_VSO1_4.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_4.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_4.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_4.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_4.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_4.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_4.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_4.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_4.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_4.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_4.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_4.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_4.0k/scores


SRC=osv
TGT=osv1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_4.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OSV_OSV1_4.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_4.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_4.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_4.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_4.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_4.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_4.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_4.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_4.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_4.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_4.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_4.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_4.0k/scores


SRC=svo
TGT=sov
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_8.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_SOV_8.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_8.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_8.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_8.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_8.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_8.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_8.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_8.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_8.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_8.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_8.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_8.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_8.0k/scores


SRC=svo
TGT=vos
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_8.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_VOS_8.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_8.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_8.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_8.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_8.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_8.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_8.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_8.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_8.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_8.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_8.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_8.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_8.0k/scores


SRC=svo
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_8.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_OVS_8.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_8.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_8.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_8.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_8.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_8.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_8.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_8.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_8.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_8.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_8.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_8.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_8.0k/scores


SRC=sov
TGT=vos
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_8.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_VOS_8.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_8.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_8.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_8.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_8.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_8.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_8.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_8.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_8.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_8.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_8.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_8.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_8.0k/scores


SRC=sov
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_8.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_OVS_8.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_8.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_8.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_8.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_8.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_8.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_8.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_8.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_8.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_8.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_8.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_8.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_8.0k/scores


SRC=vso
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_8.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_OSV_8.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_8.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_8.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_8.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_8.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_8.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_8.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_8.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_8.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_8.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_8.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_8.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_8.0k/scores


SRC=vos
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_8.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_OSV_8.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_8.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_8.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_8.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_8.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_8.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_8.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_8.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_8.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_8.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_8.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_8.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_8.0k/scores


SRC=osv
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_8.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OSV_OVS_8.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_8.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_8.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_8.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_8.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_8.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_8.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_8.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_8.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_8.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_8.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_8.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_8.0k/scores


SRC=sov
TGT=sov1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_8.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_SOV1_8.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_8.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_8.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_8.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_8.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_8.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_8.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_8.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_8.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_8.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_8.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_8.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_8.0k/scores


SRC=vos
TGT=vos1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_8.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_VOS1_8.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_8.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_8.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_8.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_8.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_8.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_8.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_8.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_8.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_8.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_8.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_8.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_8.0k/scores


SRC=ovs
TGT=ovs1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_8.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OVS_OVS1_8.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_8.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_8.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_8.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_8.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_8.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_8.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_8.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_8.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_8.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_8.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_8.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_8.0k/scores


SRC=svo
TGT=vso
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_16.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_VSO_16.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_16.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_16.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_16.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_16.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_16.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_16.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_16.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_16.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_16.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_16.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_16.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_16.0k/scores


SRC=svo
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_16.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_OSV_16.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_16.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_16.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_16.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_16.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_16.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_16.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_16.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_16.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_16.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_16.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_16.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_16.0k/scores


SRC=sov
TGT=vso
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_16.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_VSO_16.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_16.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_16.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_16.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_16.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_16.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_16.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_16.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_16.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_16.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_16.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_16.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_16.0k/scores


SRC=sov
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_16.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_OSV_16.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_16.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_16.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_16.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_16.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_16.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_16.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_16.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_16.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_16.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_16.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_16.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_16.0k/scores


SRC=vso
TGT=vos
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_16.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_VOS_16.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_16.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_16.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_16.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_16.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_16.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_16.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_16.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_16.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_16.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_16.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_16.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_16.0k/scores


SRC=vso
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_16.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_OVS_16.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_16.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_16.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_16.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_16.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_16.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_16.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_16.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_16.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_16.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_16.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_16.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_16.0k/scores


SRC=vos
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_16.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_OVS_16.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_16.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_16.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_16.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_16.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_16.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_16.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_16.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_16.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_16.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_16.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_16.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_16.0k/scores


SRC=svo
TGT=svo1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_16.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_SVO1_16.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_16.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_16.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_16.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_16.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_16.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_16.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_16.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_16.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_16.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_16.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_16.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_16.0k/scores


SRC=vso
TGT=vso1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_16.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_VSO1_16.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_16.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_16.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_16.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_16.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_16.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_16.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_16.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_16.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_16.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_16.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_16.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_16.0k/scores


SRC=osv
TGT=osv1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_16.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OSV_OSV1_16.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_16.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_16.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_16.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_16.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_16.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_16.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_16.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_16.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_16.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_16.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_16.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_16.0k/scores


SRC=svo
TGT=sov
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_32.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_SOV_32.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_32.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_32.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_32.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_32.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_32.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_32.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_32.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_32.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_32.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_32.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_32.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_32.0k/scores


SRC=svo
TGT=vos
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_32.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_VOS_32.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_32.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_32.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_32.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_32.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_32.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_32.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_32.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_32.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_32.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_32.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_32.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_32.0k/scores


SRC=svo
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_32.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_OVS_32.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_32.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_32.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_32.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_32.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_32.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_32.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_32.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_32.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_32.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_32.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_32.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_32.0k/scores


SRC=sov
TGT=vos
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_32.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_VOS_32.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_32.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_32.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_32.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_32.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_32.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_32.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_32.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_32.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_32.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_32.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_32.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_32.0k/scores


SRC=sov
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_32.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_OVS_32.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_32.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_32.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_32.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_32.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_32.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_32.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_32.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_32.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_32.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_32.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_32.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_32.0k/scores


SRC=vso
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_32.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_OSV_32.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_32.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_32.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_32.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_32.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_32.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_32.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_32.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_32.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_32.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_32.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_32.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_32.0k/scores


SRC=vos
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_32.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_OSV_32.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_32.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_32.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_32.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_32.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_32.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_32.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_32.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_32.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_32.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_32.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_32.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_32.0k/scores


SRC=osv
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_32.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OSV_OVS_32.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_32.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_32.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_32.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_32.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_32.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_32.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_32.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_32.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_32.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_32.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_32.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_32.0k/scores


SRC=sov
TGT=sov1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_32.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_SOV1_32.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_32.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_32.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_32.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_32.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_32.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_32.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_32.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_32.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_32.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_32.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_32.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_32.0k/scores


SRC=vos
TGT=vos1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_32.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_VOS1_32.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_32.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_32.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_32.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_32.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_32.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_32.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_32.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_32.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_32.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_32.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_32.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_32.0k/scores


SRC=ovs
TGT=ovs1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_32.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OVS_OVS1_32.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_32.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_32.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_32.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_32.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_32.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_32.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_32.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_32.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_32.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_32.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_32.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_32.0k/scores


SRC=svo
TGT=vso
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_64.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_VSO_64.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_64.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_64.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_64.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_64.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_64.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_64.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_64.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_64.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_64.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_64.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_64.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_64.0k/scores


SRC=svo
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_64.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_OSV_64.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_64.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_64.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_64.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_64.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_64.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_64.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_64.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_64.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_64.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_64.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_64.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_64.0k/scores


SRC=sov
TGT=vso
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_64.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_VSO_64.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_64.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_64.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_64.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_64.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_64.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_64.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_64.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_64.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_64.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_64.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_64.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_64.0k/scores


SRC=sov
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_64.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_OSV_64.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_64.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_64.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_64.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_64.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_64.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_64.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_64.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_64.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_64.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_64.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_64.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_64.0k/scores


SRC=vso
TGT=vos
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_64.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_VOS_64.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_64.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_64.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_64.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_64.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_64.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_64.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_64.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_64.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_64.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_64.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_64.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_64.0k/scores


SRC=vso
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_64.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_OVS_64.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_64.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_64.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_64.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_64.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_64.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_64.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_64.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_64.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_64.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_64.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_64.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_64.0k/scores


SRC=vos
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_64.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_OVS_64.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_64.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_64.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_64.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_64.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_64.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_64.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_64.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_64.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_64.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_64.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_64.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_64.0k/scores


SRC=svo
TGT=svo1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_64.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_SVO1_64.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_64.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_64.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_64.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_64.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_64.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_64.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_64.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_64.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_64.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_64.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_64.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_64.0k/scores


SRC=vso
TGT=vso1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_64.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_VSO1_64.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_64.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_64.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_64.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_64.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_64.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_64.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_64.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_64.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_64.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_64.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_64.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_64.0k/scores


SRC=osv
TGT=osv1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_64.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OSV_OSV1_64.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_64.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_64.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_64.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_64.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_64.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_64.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_64.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_64.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_64.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_64.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_64.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_64.0k/scores


SRC=svo
TGT=sov
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_SOV_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_128.0k/scores


SRC=svo
TGT=vos
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_VOS_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_128.0k/scores


SRC=svo
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_OVS_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_128.0k/scores


SRC=sov
TGT=vos
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_VOS_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_128.0k/scores


SRC=sov
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_OVS_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_128.0k/scores


SRC=vso
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_OSV_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_128.0k/scores


SRC=vos
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_OSV_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_128.0k/scores


SRC=osv
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OSV_OVS_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_128.0k/scores


SRC=sov
TGT=sov1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_SOV1_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_128.0k/scores


SRC=vos
TGT=vos1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_VOS1_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_128.0k/scores


SRC=ovs
TGT=ovs1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_128.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OVS_OVS1_128.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_128.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_128.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_128.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_128.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_128.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_128.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_128.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_128.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_128.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_128.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_128.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_128.0k/scores


SRC=svo
TGT=vso
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_VSO_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VSO_256.0k/scores


SRC=svo
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_OSV_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OSV_256.0k/scores


SRC=sov
TGT=vso
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_VSO_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VSO_256.0k/scores


SRC=sov
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_OSV_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OSV_256.0k/scores


SRC=vso
TGT=vos
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_VOS_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VOS_256.0k/scores


SRC=vso
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_OVS_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OVS_256.0k/scores


SRC=vos
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_OVS_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OVS_256.0k/scores


SRC=svo
TGT=svo1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_SVO1_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SVO1_256.0k/scores


SRC=vso
TGT=vso1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_VSO1_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_VSO1_256.0k/scores


SRC=osv
TGT=osv1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_256.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OSV_OSV1_256.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_256.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_256.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_256.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_256.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_256.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_256.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_256.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_256.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_256.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_256.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_256.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OSV1_256.0k/scores


SRC=svo
TGT=sov
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_512.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_SOV_512.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_512.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_512.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_512.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_512.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_512.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_512.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_512.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_512.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_512.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_512.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_512.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_SOV_512.0k/scores


SRC=svo
TGT=vos
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_512.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_VOS_512.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_512.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_512.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_512.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_512.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_512.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_512.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_512.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_512.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_512.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_512.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_512.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_VOS_512.0k/scores


SRC=svo
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_512.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SVO_OVS_512.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_512.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_512.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_512.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_512.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_512.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_512.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_512.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_512.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_512.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_512.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_512.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SVO_OVS_512.0k/scores


SRC=sov
TGT=vos
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_512.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_VOS_512.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_512.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_512.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_512.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_512.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_512.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_512.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_512.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_512.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_512.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_512.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_512.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_VOS_512.0k/scores


SRC=sov
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_512.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_OVS_512.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_512.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_512.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_512.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_512.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_512.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_512.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_512.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_512.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_512.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_512.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_512.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_OVS_512.0k/scores


SRC=vso
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_512.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VSO_OSV_512.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_512.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_512.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_512.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_512.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_512.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_512.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_512.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_512.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_512.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_512.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_512.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VSO_OSV_512.0k/scores


SRC=vos
TGT=osv
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_512.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_OSV_512.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_512.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_512.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_512.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_512.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_512.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_512.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_512.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_512.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_512.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_512.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_512.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_OSV_512.0k/scores


SRC=osv
TGT=ovs
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_512.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OSV_OVS_512.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_512.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_512.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_512.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_512.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_512.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_512.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_512.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_512.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_512.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_512.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_512.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OSV_OVS_512.0k/scores


SRC=sov
TGT=sov1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_512.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/SOV_SOV1_512.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_512.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_512.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_512.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_512.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_512.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_512.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_512.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_512.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_512.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_512.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_512.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/SOV_SOV1_512.0k/scores


SRC=vos
TGT=vos1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_512.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/VOS_VOS1_512.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_512.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_512.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_512.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_512.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_512.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_512.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_512.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_512.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_512.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_512.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_512.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/VOS_VOS1_512.0k/scores


SRC=ovs
TGT=ovs1
MAX_EPOCHS=750

mkdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_512.0k
cp -R /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/data/OVS_OVS1_512.0k /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_512.0k/data

SCRIPT_PATH=/mnt/storage/jcheigh/testperanto/appa-mt/fairseq
bash $SCRIPT_PATH/prepare-data.sh /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_512.0k $SRC $TGT /mnt/storage/jcheigh/testperanto/appa-mt/fairseq

TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_512.0k/data-tokenized
BINARY_TEXT=/mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_512.0k/data-bin

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
    --save-dir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_512.0k/checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 30 \
    --tensorboard-logdir /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_512.0k/tensorboard_logs/

fairseq-generate $BINARY_TEXT \
    --path /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_512.0k/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_512.0k/translations

python $SCRIPT_PATH/extract_hyp_and_ref.py /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_512.0k/translations

sacrebleu /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_512.0k/translations.ref -i /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_512.0k/translations.hyp -m bleu chrf > /mnt/storage/jcheigh/testperanto/experiment_pipeline/experiment_data/experiments/svo_permutations/results/OVS_OVS1_512.0k/scores

