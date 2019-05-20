#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=/home/usuaris/veu/jordi.armengol/tfg/new/logs/flores/en-ne/align_tags1.log


FEATURE_TAGGER_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/src/fairseq-baseline-factored/preprocessing/stanford"
PYTHON="python"

# Activate conda environment
source ~/.bashrc
conda activate env

stdbuf -i0 -e0 -o0 $PYTHON $FEATURE_TAGGER_DIR/align_flores_sentencepiece.py