#!/usr/bin/env bash

git clone https://github.com/facebookresearch/flores.git

source ~/.bashrc
conda activate env

pip install sacrebleu sentencepiece

cd flores
bash download-data.sh

bash prepare-neen.sh
bash prepare-sien.sh

cp -r data /home/usuaris/veu/jordi.armengol/tfg/new/data/flores
cp -r data-bin /home/usuaris/veu/jordi.armengol/tfg/new/data/flores
