#!/bin/bash


#SBATCH -p veu # Partition to submit to
#S BATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/jordi.armengol/tfg/new/logs/new_islt14_scripts/generate29-average-lemmasS-10k-one-sum.log


WORKING_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14-preprocessed-joined-stanford"
#SRC="de"
#TGT="en"
DEST_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14-preprocessed-joined-stanford"
CP_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14-preprocessed-joined-stanford/checkpoints31"
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
	--beam 5 --batch-size 1 --lang-pairs de_tokensS-en,de_tokensS_lemmas-en --task factored_translation --remove-bpe --target-lang en --multiple-encoders False
