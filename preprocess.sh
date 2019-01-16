#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=preprocess-enes.log


WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
SRC="en"
TGT="es"

TRN_PREF="general.es-en.tc.clean"
VAL_PREF="newstest2012.tc"
TES_PREF="newstest2013.tc"
PYTHON="/home/usuaris/veu/cescola/virtualenv-16.0.0/torch/bin/python"
FAIRSEQ_DIR="/home/usuaris/veu/cescola/fairseq"

DEST_DIR="data-bin/wmt13.tokenized.32k.en-es"

mkdir $DEST_DIR


N_OP=32000


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

stdbuf -i0 -e0 -o0  $PYTHON $FAIRSEQ_DIR/preprocess.py --source-lang $SRC --target-lang $TGT \
    --trainpref $DEST_DIR/${TRN_PREF}.bpe --validpref $DEST_DIR/${VAL_PREF}.bpe --testpref $DEST_DIR/${TES_PREF}.bpe \
    --destdir $DEST_DIR  --nwordstgt 32000 --nwordssrc 32000

#Copy test for autoencoder translation

cp $DEST_DIR/test.${SRC}-${TGT}.${SRC}.bin  $DEST_DIR/test.${SRC}-${SRC}.${SRC}.bin
cp $DEST_DIR/test.${SRC}-${TGT}.${SRC}.idx  $DEST_DIR/test.${SRC}-${SRC}.${SRC}.idx
cp $DEST_DIR/test.${SRC}-${TGT}.${TGT}.bin  $DEST_DIR/test.${TGT}-${TGT}.${TGT}.bin
cp $DEST_DIR/test.${SRC}-${TGT}.${TGT}.idx  $DEST_DIR/test.${TGT}-${TGT}.${TGT}.idx


