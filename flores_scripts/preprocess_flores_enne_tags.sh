#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=/home/usuaris/veu/jordi.armengol/tfg/new/logs/flores/en-ne/preprocess2_tags.log

BPESIZE=5000
WORKING_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/flores/data/wiki_ne_en_bpe${BPESIZE}"
SRC="en"
TGT="ne"


TRN_PREF="train"
VAL_PREF="valid"
TES_PREF="test"
PYTHON="python"

FAIRSEQ_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/src/fairseq-baseline-factored"

#DEST_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/flores/data-bin/wiki_${SRC}_${TGT}_bpe${BPESIZE}"
DEST_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/flores/data-bin/wiki_ne_en_bpe5000"


# Activate conda environment
source ~/.bashrc
conda activate env

SRC_TAG="en_lemmas"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC_TAG --target-lang $TGT \
    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR --workers 4 --tgtdict $DEST_DIR/dict.ne.txt

SRC_TAG="en_synsets"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC_TAG --target-lang $TGT \
    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --workers 4  --tgtdict $DEST_DIR/dict.ne.txt

SRC_TAG="en_pos"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC_TAG --target-lang $TGT \
    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --workers 4  --tgtdict $DEST_DIR/dict.ne.txt

SRC_TAG="en_deps"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC_TAG --target-lang $TGT \
    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --workers 4  --tgtdict $DEST_DIR/dict.ne.txt

SRC_TAG="en_tags"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC_TAG --target-lang $TGT \
    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --workers 4  --tgtdict $DEST_DIR/dict.ne.txt

SRC_TAG="en_subword_tags"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC_TAG --target-lang $TGT \
    --trainpref $WORKING_DIR/${TRN_PREF}.bpe --validpref $WORKING_DIR/${VAL_PREF}.bpe --testpref $WORKING_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR   --workers 4  --tgtdict $DEST_DIR/dict.ne.txt