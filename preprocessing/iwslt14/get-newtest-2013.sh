#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt
BPE_TOKENS=10000


URL="https://wit3.fbk.eu/archive/2016-01/texts/de/en/de-en.tgz"
GZ=de-en13.tgz

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=de
tgt=en
lang=de-en
prep=iwslt13.tokenized.de-en
tmp=$prep/tmp13
orig=orig13

mkdir -p $orig $tmp $prep

echo "Downloading data from ${URL}..."
cd $orig
wget -O $GZ "$URL"

if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf $GZ
cd ..

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT16.TED*.$l.xml`; do
    fname=${o##*/}
    f=$tmp/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $f
    echo ""
    done
done


echo "creating test..."
for l in $src $tgt; do
    #awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/valid.$l
    #awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/train.$l

    cat $tmp/IWSLT16.TED.tst2013.de-en.$l > $tmp/test.$l
done


TRAIN=$tmp/train.en-de
BPE_CODE=$prep/code

mv orig13 ../../../../data/
mv iwslt13.tokenized.de-en ../../../../data/
