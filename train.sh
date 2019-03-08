#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=../../logs/train-baseline-joined-bpe.log


WORKING_DIR="../../data/iwslt14-preprocessed-joined"
CP_DIR="../../data/iwslt14-preprocessed-joined/checkpoints"
PYTHON="python"
FAIRSEQ_DIR="fairseq"
#SAVE_UPDATES=5000

source ~/.bashrc
conda activate env

mkdir -p $CP_DIR

#stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/train.py $WORKING_DIR \
# --arch  transformer_iwslt_de_en  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.001 --min-lr 1e-09 --dropout 0.1 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 3584 --update-freq 16 --save-dir $CP_DIR --save-interval-updates $SAVE_UPDATES --source-lang de --target-lang en

stdbuf -i0 -e0 -o0 $PYTHON $FAIRSEQ_DIR/train.py $WORKING_DIR \
 --arch  transformer_iwslt_de_en  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.001 --min-lr 1e-09 --dropout 0.1 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 3584 --update-freq 16 --save-dir $CP_DIR --source-lang de --target-lang en --max-epoch 100
