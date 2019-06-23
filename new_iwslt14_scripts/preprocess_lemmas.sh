#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=/home/usuaris/veu/jordi.armengol/tfg/new/logs/new_islt14_scripts/preprocess-lemmas10k.log


WORKING_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14.tokenized.de-en/tmp"

TRN_PREF="train"
VAL_PREF="valid"
TES_PREF="test"
PYTHON="python"

FAIRSEQ_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/src/fairseq-baseline-factored"

DEST_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14-preprocessed-joined"

# Activate conda environment
source ~/.bashrc
conda activate env

SRC="de_lemmas10k"
TGT="en"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --tgtdict $DEST_DIR/dict.en.txt --nwordssrc 10000
