#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/jordi.armengol/tfg/new/logs/generate12-factored-bpe-postags-remove-bpe-al-at-reduced-freeze-7.log


WORKING_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14-preprocessed-joined"
SRC="de"
TGT="en"
DEST_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14-preprocessed-joined"
CP_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14-preprocessed-joined/checkpoints12"
#CP="checkpoint_best.pt"
CP="model.pt"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/src/fairseq-baseline-factored"

# Activate conda environment
source ~/.bashrc
conda activate env


#stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/generate.py $DEST_DIR --path $CP_DIR/$CP \
#    --beam 5 --batch-size 1 --source-lang ${SRC} --target-lang ${TGT} --task translation --remove-bpe

stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/generate.py $DEST_DIR --path $CP_DIR/$CP \
	--beam 5 --batch-size 1 --lang-pairs de-en,de_postags_at-en --task factored_translation --remove-bpe
