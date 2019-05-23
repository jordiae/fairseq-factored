#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=/home/usuaris/veu/jordi.armengol/tfg/new/logs/flores/en-ne/assign_align_synsets1.log


FLORES_SYNSET_TAGGER_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/src/fairseq-baseline-factored/preprocessing/flores"
PYTHON="python"

# Activate conda environment
source ~/.bashrc
conda activate env

stdbuf -i0 -e0 -o0 $PYTHON FLORES_SYNSET_TAGGER_DIR/assign_align_sentencepiece_synsets.py
