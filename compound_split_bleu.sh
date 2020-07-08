#!/bin/bash

GEN=$1

SYS=$GEN.sys
REF=$GEN.ref
BLEU=$GEN.bleu

grep ^H $GEN | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $SYS
grep ^T $GEN | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $REF
python score.py --sys $SYS --ref $REF | tee $BLEU