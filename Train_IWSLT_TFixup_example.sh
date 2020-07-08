#!/usr/bin/env bash

cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

TEXT=examples/translation/iwslt14.tokenized.de-en
python preprocess.py --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20

python train.py "data-bin/iwslt14.tokenized.de-en" \
  -a "transformer_iwslt_de_en" \
  -s de \
  -t en \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --lr-scheduler inverse_sqrt_nowarmup \
  --lr 0.0005 \
  --decay-period 4000 \
  --dropout 0.5 \
  --weight-decay 0.0001 \
  --max-tokens 4000 \
  --max-epoch 300 \
  --dont-use-layernorm \
  --Tfixup \
  --share-decoder-input-output-embed \
  --fp16 \
  --tensorboard-logdir ./results/IWSLT/tensorboard/Tfixup_d5L5_cleanscheduler_fp16/ \
  --save-dir ./results/IWSLT/checkpoints/Tfixup_d5L5_cleanscheduler_fp16/ \
  --eval-bleu \
  --eval-bleu-args '{"beam": 4, "lenpen": 0.3}' \
  --eval-bleu-remove-bpe \
  --eval-bleu-detok moses \

python InferenceIWSLT.py Tfixup_d5L5_cleanscheduler_fp16 250 300