<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

## ICML'20 Improving Transformer Optimization Through Better Initialization [[paper](http://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf)]

Authors: Xiao Shi Huang, Felipe Perez, [Jimmy Ba](https://jimmylba.github.io), [Maksims Volkovs](http://www.cs.toronto.edu/~mvolkovs)

<a name="intro"/>

## Introduction
This repository contains full implementation of the T-Fixup algorithm implemented on the fairseq library, and includes both training and evaluation routines on the IWSLT'14 De-En dataset.

<a name="env"/>

## Environment
The python code is developed and tested on the following environment:
* Python 3.7
* Pytorch 1.2.0

Experiments on IWSLT'14 De-En and En-De datasets were run on NVIDIA V100 GPU with 32GB GPU memory; all other experiments were run on the IBM server with 160 POWER9 CPUs, 600GB RAM and 4 Tesla V100 GPUs

<a name="dataset"/>

## Dataset

The example execution script `Train_IWSLT_TFixup_example.sh` builds the IWSLT'14 De-En dataset; for the WMT'14 En-De and WMT'17 En-De datasets refer to the fairseq's instructions [here](https://github.com/pytorch/fairseq/tree/master/examples/translation) 

## Running The Code

1. `./Train_IWSLT_TFixup_example.sh`
2. (Optionally) launch tensorboard to monitor progress by `tensorboard --logdir=<log_path>`

This script runs the small 512-1024-4 Transformer encoder-decoder model (see paper for details) with both layer normalization and learning rate warmup removed. Starting learning rate is set to the post warmup value of 0.0005 (vs 1e-07 with warmup). By default all avialable GPUs are used, but parameters such as batchsize are set for for 1 GPU. If multiple GPUs are avaialbe, either point the script to only one GPU or adjust model parameters accordingly.

## Validation Curves
Training and validation loss curves for the T-Fixup model on IWSLT'14 De-En dataset for the first 300 epochs. One epoch is around 
1100 updates and we checkpoint the model after each epoch.
<p align="center">
<img src="https://github.com/layer6ai-labs/T-Fixup/blob/master/TFixup_IWSLT14_LossCurve.png" width="500">
</p>
BLEU score, evaluated using the average of 10 checkpoints, reaches 35.73 at epoch 278-287
