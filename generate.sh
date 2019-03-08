#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=../../logs/generate-baseline-test-joined-bpe-de2en-best-remove-bpe.log


WORKING_DIR="../../data/iwslt14-preprocessed-joined"
SRC="de"
TGT="en"
DEST_DIR="../../data/iwslt14-preprocessed-joined"
CP_DIR="../../data/iwslt14-preprocessed-joined/checkpoints"
CP="checkpoint_best.pt"
PYTHON="python"
FAIRSEQ_DIR="fairseq"

stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/generate.py $DEST_DIR --path $CP_DIR/$CP \
    --beam 5 --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation --remove-bpe