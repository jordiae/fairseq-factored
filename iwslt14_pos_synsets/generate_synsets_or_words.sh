#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/jordi.armengol/tfg/new/logs/generate19-baseline-bpe-synsets-or-words.log


WORKING_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14-preprocessed-joined"
SRC="de_synsets_or_words"
TGT="en"
DEST_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14-preprocessed-joined"
CP_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14-preprocessed-joined/checkpoints19"
CP="checkpoint_best.pt"
#CP="model.pt"
PYTHON="/home/usuaris/veu/cescola/virtualenv-16.0.0/torch/bin/python"
FAIRSEQ_DIR="/home/usuaris/veu/cescola/fairseq"

#stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/generate.py $DEST_DIR --path $CP_DIR/$CP \
#    --beam 5 --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation --remove-bpe

stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/generate.py $DEST_DIR --path $CP_DIR/$CP \
	--beam 5 --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation --remove-bpe
