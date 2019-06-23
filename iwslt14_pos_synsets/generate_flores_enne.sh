#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/jordi.armengol/tfg/new/logs/flores/en-ne/generate1.log


WORKING_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/flores/data-bin/wiki_ne_en_bpe5000"
SRC="en"
TGT="ne"
DEST_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/flores/data-bin/wiki_ne_en_bpe5000"
CP_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/flores/data-bin/wiki_ne_en_bpe5000/checkpoints_enne"
CP="checkpoint_best.pt"
#CP="model.pt"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/src/fairseq-baseline-factored"

# Activate conda environment
source ~/.bashrc
conda activate env

stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/generate.py $DEST_DIR --path $CP_DIR/$CP \
	--beam 5 --lenpen 1.2 --gen-subset valid --source-lang ${SRC} --target-lang ${TGT} --task translation --remove-bpe=sentencepiece
