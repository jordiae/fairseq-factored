#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/jordi.armengol/tfg/new/logs/new_islt14_scripts/newtest-preprocess-joined-bpe-stanfordS.log

BPE_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14-preprocessed-joined-stanford"
WORKING_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt16.tokenized.de-en/tmp16"
SRC="de_tokensS"
TGT="en"

TRN_PREF="train"
VAL_PREF="valid"
TES_PREF="test"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/src/fairseq-baseline-factored"

DEST_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt16-preprocessed-joined-stanford"

mkdir -p $DEST_DIR


# Activate conda environment
source ~/.bashrc
conda activate env

echo "apply joined bpe"

subword-nmt apply-bpe -c ${BPE_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${BPE_DIR}/${TRN_PREF}.vocab.${SRC} --vocabulary-threshold 50 < ${WORKING_DIR}/${TES_PREF}.${SRC} > ${DEST_DIR}/${TES_PREF}.bpe.${SRC}
subword-nmt apply-bpe -c ${BPE_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${BPE_DIR}/${TRN_PREF}.vocab.${TGT} --vocabulary-threshold 50 < ${WORKING_DIR}/${TES_PREF}.${TGT} > ${DEST_DIR}/${TES_PREF}.bpe.${TGT}

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
    --testpref $DEST_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --nwordstgt 32000 --nwordssrc 32000 --srcdict $BPE_DIR/dict.de_tokensS.txt --tgtdict $BPE_DIR/dict.en.txt

