#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=/home/usuaris/veu/jordi.armengol/tfg/new/logs/new_islt14_scripts/preprocess-tags.log


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

SRC="de_lemmas"
TGT="en"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --tgtdict $DEST_DIR/dict.en.txt --thresholdsrc 30

SRC="de_pos"
TGT="en"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --tgtdict $DEST_DIR/dict.en.txt --thresholdsrc 1

SRC="de_deps"
TGT="en"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --tgtdict $DEST_DIR/dict.en.txt --thresholdsrc 1

SRC="de_tags"
TGT="en"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --tgtdict $DEST_DIR/dict.en.txt --thresholdsrc 1

#SRC="de_subword_tags"
#TGT="en"

#stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
#    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
#    --destdir $DEST_DIR  --tgtdict $DEST_DIR/dict.en.txt --thresholdsrc 1
