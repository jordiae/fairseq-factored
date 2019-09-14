#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=/home/usuaris/veu/jordi.armengol/tfg/new/logs/flores/en-ne/nosp_prepare_preprocess.log

BPESIZE=5000
WORKING_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/flores/data/nosp_wiki_ne_en_bpe${BPESIZE}"
SRC="en"
TGT="ne"

TRN_PREF="train"
VAL_PREF="valid"
TES_PREF="test"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/src/fairseq-baseline-factored"

DEST_DIR=$WORKING_DIR

mkdir $DEST_DIR


N_OP=5000



# Activate conda environment
source ~/.bashrc
conda activate env

echo "apply joined bpe"

subword-nmt learn-joint-bpe-and-vocab --input ${WORKING_DIR}/${TRN_PREF}.${SRC} ${WORKING_DIR}/${TRN_PREF}.${TGT} -s $N_OP -o ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --write-vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${SRC} ${DEST_DIR}/${TRN_PREF}.vocab.${TGT}

subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${SRC} < ${WORKING_DIR}/${TRN_PREF}.${SRC} > ${DEST_DIR}/${TRN_PREF}.bpe.${SRC}
subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${TGT} --vocabulary-threshold 50 < ${WORKING_DIR}/${TRN_PREF}.${TGT} > ${DEST_DIR}/${TRN_PREF}.bpe.${TGT}

subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${SRC} < ${WORKING_DIR}/${VAL_PREF}.${SRC} > ${DEST_DIR}/${VAL_PREF}.bpe.${SRC}
subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${TGT} < ${WORKING_DIR}/${VAL_PREF}.${TGT} > ${DEST_DIR}/${VAL_PREF}.bpe.${TGT}


subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${SRC} < ${WORKING_DIR}/${TES_PREF}.${SRC} > ${DEST_DIR}/${TES_PREF}.bpe.${SRC}
subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${TGT} < ${WORKING_DIR}/${TES_PREF}.${TGT} > ${DEST_DIR}/${TES_PREF}.bpe.${TGT}


stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
    --trainpref $DEST_DIR/${TRN_PREF}.bpe --validpref $DEST_DIR/${VAL_PREF}.bpe --testpref $DEST_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --nwordstgt 5000 --nwordssrc 5000 --joined-dictionary