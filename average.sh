#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:0
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/jordi.armengol/tfg/new/logs/average12.log


WORKING_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14-preprocessed-joined"
DEST_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14-preprocessed-joined"
CP_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14-preprocessed-joined/checkpoints12"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/src/fairseq-baseline-factored"

source ~/.bashrc
conda activate env


stdbuf -i0 -e0 -o0 $PYTHON  $FAIRSEQ_DIR/scripts/average_checkpoints.py --inputs $CP_DIR \
	--num-epoch-checkpoints 10 --output $CP_DIR/model.pt
