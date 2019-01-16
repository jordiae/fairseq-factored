#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=generate-es2en.log


WORKING_DIR="/veu4/usuaris31/mruiz/large-projections/corpus/"
SRC="es"
TGT="en"
DEST_DIR="../interlingua-fairseq/data-bin/wmt13.tokenized.32k.en-es"
CP_DIR="checkpoints/esen-es2en/"
CP="checkpoint_best.pt"
PYTHON="/home/usuaris/veu/cescola/virtualenv-16.0.0/torch/bin/python"
FAIRSEQ_DIR="/home/usuaris/veu/cescola/fairseq"

stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/generate.py $DEST_DIR --path $CP_DIR/$CP \
    --beam 5 --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation
