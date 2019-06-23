#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/jordi.armengol/tfg/new/logs/new_islt14_scripts/train34_multiple_concat_lemmas10_stanfordS.log

WORKING_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14-preprocessed-joined-stanford"
CP_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14-preprocessed-joined-stanford/checkpoints34"
PYTHON="python"
FAIRSEQ_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/src/fairseq-baseline-factored"

source ~/.bashrc
conda activate env

mkdir -p $CP_DIR

#stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/train.py $WORKING_DIR \
# --task factored_translation --arch factored_transformer_iwslt_de_en  --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 --min-lr 1e-09 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4000  --save-dir $CP_DIR --lang-pairs de-en,de_postags_at-en --max-update 50000 --factors-to-freeze de_postags_at-en --freeze-factors-epoch 7

stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/train.py $WORKING_DIR \
 --task factored_translation --arch factored_transformer__iwslt_de_en_lemmas_stanfordS  --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 --min-lr 1e-09 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4000  --save-dir $CP_DIR --lang-pairs de_tokensS-en,de_tokensS_lemmas-en --max-update 50000 --multiple-encoders True --keep-last-epochs 2 
