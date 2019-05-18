#!/bin/bash


#SBATCH -p veu-fast # Partition to submit to
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=/home/usuaris/veu/jordi.armengol/tfg/new/logs/flores/en-ne/feature_tagger1.log


FEATURE_TAGGER_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/src/fairseq-baseline-factored/preprocessing/stanford"
PYTHON="python"

# Activate conda environment
source ~/.bashrc
conda activate env

pip install spacy-stanfordnlp

mkdir /home/usuaris/veu/jordi.armengol/stanfordnlp_resources
wget https://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/en_ewt_models.zip/ -P /home/usuaris/veu/jordi.armengol/stanfordnlp_resources
unzip /home/usuaris/veu/jordi.armengol/stanfordnlp_resources/en_ewt_models.zip

stdbuf -i0 -e0 -o0 $PYTHON $FEATURE_TAGGER_DIR/feature_tagger.py