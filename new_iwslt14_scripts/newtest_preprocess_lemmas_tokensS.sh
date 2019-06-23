#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=/home/usuaris/veu/jordi.armengol/tfg/new/logs/new_islt14_scripts/newtest-lemmas-preprocess-lemmas10k-tokensS.log


WORKING_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt16.tokenized.de-en/tmp16"

TRN_PREF="train"
VAL_PREF="valid"
TES_PREF="test"
PYTHON="python"

FAIRSEQ_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/src/fairseq-baseline-factored"
DICT_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14-preprocessed-joined-stanford"
DEST_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt16-preprocessed-joined-stanford"

# Activate conda environment
source ~/.bashrc
conda activate env

SRC="de_tokensS_lemmas"
TGT="en"

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
     --testpref $WORKING_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --srcdict $DICT_DIR/dict.de_tokensS_lemmas.txt --tgtdict $DICT_DIR/dict.en.txt
