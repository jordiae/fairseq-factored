#!/usr/bin/env bash

git clone https://github.com/facebookresearch/flores.git

source ~/.bashrc
conda activate env

pip install sacrebleu sentencepiece

bash flores/download-data.sh