import sentencepiece as spm
import os

FLORES_DATA_PATH = '/home/usuaris/veu/jordi.armengol/tfg/new/data/flores'
BPE_MODEL_PATH = os.path.join(FLORES_DATA_PATH, 'data-bin', 'wiki_ne_en_bpe5000', 'sentencepiece.bpe.model')
NEEN_DATA_PATH = os.path.join(FLORES_DATA_PATH, 'data', 'wiki_ne_en_bpe5000')
s = spm.SentencePieceProcessor()
s.Load(BPE_MODEL_PATH)
train_en = open(os.path.join(NEEN_DATA_PATH, 'train.en'), 'r').read()
train_en_bpe = open(os.path.join(NEEN_DATA_PATH, 'train.bpe.en'), 'r').read()
train_en_l = train_en.splitlines()
train_en_bpe_l = train_en_bpe.splitlines()
train_ne = open(os.path.join(NEEN_DATA_PATH, 'train.ne'), 'r').read()
train_ne_bpe = open(os.path.join(NEEN_DATA_PATH, 'train.bpe.ne'), 'r').read()
train_ne_l = train_ne.splitlines()
train_ne_bpe_l = train_ne_bpe.splitlines()

min_len = 1
max_len = 250


def encode(x):
    return s.EncodeAsPieces(x)


def valid(line):
            return (
                (min_len is None or len(line) >= min_len)
                and (max_len is None or len(line) <= max_len)
            )


def encode_line(line):
    line = line.strip()
    if len(line) > 0:
        line = encode(line)
        if valid(line):
            return line
        else:
            pass
    else:
        pass
    return None

removed_sentences = []
for i, lines in enumerate(zip(train_en_l, train_ne_l)):
    enc_lines = list(map(encode_line, lines))
    if not any(enc_line is None for enc_line in enc_lines):
        pass
    else:
        removed_sentences.append(i)

print(len(removed_sentences), 'sentences to remove')

new_train_en_l = []
new_train_ne_l = []

for index, line in enumerate(zip(train_en_l, train_ne_l)):
    if index not in removed_sentences:
        new_train_en_l.append(line[0])
        new_train_ne_l.append(line[1])


def reverse_splitlines(lines):
    s = ''
    for l in lines:
        s += l + '\n'
    return s


new_train_en = reverse_splitlines(new_train_en_l)
new_train_ne = reverse_splitlines(new_train_ne_l)

with open(os.path.join(NEEN_DATA_PATH, 'train.en'), 'w') as file:
    file.write(new_train_en)

with open(os.path.join(NEEN_DATA_PATH, 'train.ne'), 'w') as file:
    file.write(new_train_ne)
