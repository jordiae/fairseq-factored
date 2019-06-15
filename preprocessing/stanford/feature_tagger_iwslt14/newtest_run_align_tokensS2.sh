#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=/home/usuaris/veu/jordi.armengol/tfg/new/logs/new_islt14_scripts/newtest_align_features_de_tokensS2.log


FEATURE_TAGGER_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/src/fairseq-baseline-factored/preprocessing/stanford/feature_tagger_iwslt14"
PYTHON="python"

# Activate conda environment
source ~/.bashrc
conda activate env

stdbuf -i0 -e0 -o0 $PYTHON $FEATURE_TAGGER_DIR/newtest_align_features_iwslt14_bpe_tokensS2.py
