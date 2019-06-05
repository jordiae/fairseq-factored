#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH -x veuc09,veuc06
#SBATCH --gres=gpu:1
#SBATCH --export LD_PRELOAD="/home/usuaris/veu/jordi.armengol/tfg/new/src/fairseq-baseline-factored/preprocessing/stanford/feature_tagger_iwslt14/libtcmalloc_minimal.so.4"
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=/home/usuaris/veu/jordi.armengol/tfg/new/logs/feature_tagger_de1_test.log


FEATURE_TAGGER_DIR="/home/usuaris/veu/jordi.armengol/tfg/new/src/fairseq-baseline-factored/preprocessing/stanford/feature_tagger_iwslt14"
PYTHON="python"

# Activate conda environment
source ~/.bashrc
conda activate env
#export LD_PRELOAD="/home/usuaris/veu/jordi.armengol/tfg/new/src/fairseq-baseline-factored/preprocessing/stanford/feature_tagger_iwslt14/libtcmalloc_minimal.so.4"
#source /home/usuaris/veu/cescola/virtualenv-16.0.0/stanford-tf/bin/activate
#pip install spacy-stanfordnlp

#mkdir /home/usuaris/veu/jordi.armengol/stanfordnlp_resources
#wget https://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/en_ewt_models.zip -P /home/usuaris/veu/jordi.armengol/stanfordnlp_resources
#http://nlp.stanford.edu/software/stanfordnlp_models/0.2.0/de_gsd_models.zip
#cd /home/usuaris/veu/jordi.armengol/stanfordnlp_resources
#unzip de_gsd_models.zip
echo $LD_PRELOAD
stdbuf -i0 -e0 -o0 $PYTHON $FEATURE_TAGGER_DIR/feature_tagger.py
