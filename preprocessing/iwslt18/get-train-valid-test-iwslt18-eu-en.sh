#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh
# set -e

#echo 'Cloning Moses github repository (for tokenization scripts)...'
#git clone https://github.com/moses-smt/mosesdecoder.git

#echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
#git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
#BPEROOT=subword-nmt
#BPE_TOKENS=10000

URL="https://wit3.fbk.eu/archive/2018-01/texts/eu/en/eu-en.tgz"
GZ=eu-en.tgz
# Train/valid: https://wit3.fbk.eu/archive/2018-01/texts/eu/en/eu-en.tgz
# Test: https://wit3.fbk.eu/archive/2018-01/evaluation_sets/IWSLT18.LowResourceMT.tst.srcOnly.tgz

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=eu
tgt=en
lang=eu-en
prep=iwslt18.tokenized.eu-en
tmp=$prep/tmp
orig=orig18

mkdir -p $orig $tmp $prep

echo "Downloading data from ${URL}..."
cd $orig
wget "$URL"

if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf $GZ
cd ..

cp -r ${orig}/IWSLT18.LowResourceMT.train_dev/eu-en ${orig}/eu-en

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat $orig/$lang/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    sed -e '/^</d' | \
    sed -e 's/^TED Talk Subtitles and Transcript: //' | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done
perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175
for l in $src $tgt; do
    perl $LC < $tmp/train.tags.$lang.clean.$l > $tmp/train.tags.$lang.$l
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT18.TED*.$l.xml`; do
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


echo "creating train, valid, test..."
for l in $src $tgt; do
    awk '{if (NR%4 == 0)  print $0; }' $tmp/train.tags.eu-en.$l > $tmp/valid.$l
    awk '{if (NR%4 != 0)  print $0; }' $tmp/train.tags.eu-en.$l > $tmp/train.$l

    cat $tmp/IWSLT18.TED.dev2018.eu-en.$l \
        > $tmp/test.$l
done

:'
TRAIN=$tmp/train.en-eu
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done
'

mkdir ../../../../data/iwslt18
mv orig18 ../../../../data/iwslt18
mv iwslt18.tokenized.eu-en ../../../../data/iwslt18
