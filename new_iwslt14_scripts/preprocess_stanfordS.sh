#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/jordi.armengol/tfg/new/logs/new_islt14_scripts/preprocess-joined-bpe-stanfordS.log


WORKING_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14.tokenized.de-en.stanford/tmp"
SRC="de_tokensS"
TGT="en"

TRN_PREF="train"
VAL_PREF="valid"
TES_PREF="test"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/src/fairseq-baseline-factored"

DEST_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14-preprocessed-joined-stanford"

mkdir $DEST_DIR


N_OP=10000



# Activate conda environment
source ~/.bashrc
conda activate env

echo "apply joined bpe"

subword-nmt learn-joint-bpe-and-vocab --input ${WORKING_DIR}/${TRN_PREF}.${SRC} ${WORKING_DIR}/${TRN_PREF}.${TGT} -s $N_OP -o ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --write-vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${SRC} ${DEST_DIR}/${TRN_PREF}.vocab.${TGT}

subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${SRC} --vocabulary-threshold 50 < ${WORKING_DIR}/${TRN_PREF}.${SRC} > ${DEST_DIR}/${TRN_PREF}.bpe.${SRC} 
subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${TGT} --vocabulary-threshold 50 < ${WORKING_DIR}/${TRN_PREF}.${TGT} > ${DEST_DIR}/${TRN_PREF}.bpe.${TGT}

subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${SRC} --vocabulary-threshold 50 < ${WORKING_DIR}/${VAL_PREF}.${SRC} > ${DEST_DIR}/${VAL_PREF}.bpe.${SRC}
subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${TGT} --vocabulary-threshold 50 < ${WORKING_DIR}/${VAL_PREF}.${TGT} > ${DEST_DIR}/${VAL_PREF}.bpe.${TGT}


subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${SRC} --vocabulary-threshold 50 < ${WORKING_DIR}/${TES_PREF}.${SRC} > ${DEST_DIR}/${TES_PREF}.bpe.${SRC}
subword-nmt apply-bpe -c ${DEST_DIR}/${TRN_PREF}.codes.${SRC}-${TGT} --vocabulary ${DEST_DIR}/${TRN_PREF}.vocab.${TGT} --vocabulary-threshold 50 < ${WORKING_DIR}/${TES_PREF}.${TGT} > ${DEST_DIR}/${TES_PREF}.bpe.${TGT}

:'

echo "apply bpe to " $SRC

subword-nmt learn-bpe -s $N_OP < ${WORKING_DIR}/${TRN_PREF}.${SRC} > ${DEST_DIR}/${TRN_PREF}.codes.${SRC}
subword-nmt apply-bpe -c  ${DEST_DIR}/${TRN_PREF}.codes.${SRC} < ${WORKING_DIR}/${TRN_PREF}.${SRC} >  ${DEST_DIR}/${TRN_PREF}.bpe.${SRC}
subword-nmt apply-bpe -c  ${DEST_DIR}/${TRN_PREF}.codes.${SRC} < ${WORKING_DIR}/${VAL_PREF}.${SRC} >  ${DEST_DIR}/${VAL_PREF}.bpe.${SRC}
subword-nmt apply-bpe -c  ${DEST_DIR}/${TRN_PREF}.codes.${SRC} < ${WORKING_DIR}/${TES_PREF}.${SRC} >  ${DEST_DIR}/${TES_PREF}.bpe.${SRC}

echo "apply bpe to " $TGT

subword-nmt learn-bpe -s $N_OP < ${WORKING_DIR}/${TRN_PREF}.${TGT} > ${DEST_DIR}/${TRN_PREF}.codes.${TGT}
subword-nmt apply-bpe -c  ${DEST_DIR}/${TRN_PREF}.codes.${SRC} < ${WORKING_DIR}/${TRN_PREF}.${TGT} >  ${DEST_DIR}/${TRN_PREF}.bpe.${TGT}
subword-nmt apply-bpe -c  ${DEST_DIR}/${TRN_PREF}.codes.${SRC} < ${WORKING_DIR}/${VAL_PREF}.${TGT} >  ${DEST_DIR}/${VAL_PREF}.bpe.${TGT}
subword-nmt apply-bpe -c  ${DEST_DIR}/${TRN_PREF}.codes.${SRC} < ${WORKING_DIR}/${TES_PREF}.${TGT} >  ${DEST_DIR}/${TES_PREF}.bpe.${TGT}
'
stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
    --trainpref $DEST_DIR/${TRN_PREF}.bpe --validpref $DEST_DIR/${VAL_PREF}.bpe --testpref $DEST_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --nwordstgt 32000 --nwordssrc 32000

#Copy test for autoencoder translation

cp $DEST_DIR/test.${SRC}-${TGT}.${SRC}.bin  $DEST_DIR/test.${SRC}-${SRC}.${SRC}.bin
cp $DEST_DIR/test.${SRC}-${TGT}.${SRC}.idx  $DEST_DIR/test.${SRC}-${SRC}.${SRC}.idx
cp $DEST_DIR/test.${SRC}-${TGT}.${TGT}.bin  $DEST_DIR/test.${TGT}-${TGT}.${TGT}.bin
cp $DEST_DIR/test.${SRC}-${TGT}.${TGT}.idx  $DEST_DIR/test.${TGT}-${TGT}.${TGT}.idx

