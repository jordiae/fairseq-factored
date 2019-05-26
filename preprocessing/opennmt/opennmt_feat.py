import os
LANG = 'de'
PATH = '/home/usuaris/veu/jordi.armengol/tfg/new/OpenNMT-py/data/iwslt14'


def format_opennmt(text_bpe, text_lemma, text_pos, text_dep, text_tag):
    sep = '\uFFE8'
    res = ''
    for (index, (line_bpe, line_lemma, line_pos, line_dep, line_tag)) in enumerate(zip(text_bpe.splitlines(), text_lemma.splitlines(), text_pos.splitlines(), text_dep.splitlines(), text_tag.splitlines())):
        for bpe, lemma, pos, dep, tag in zip(line_bpe.split(), line_lemma.split(), line_pos.split(), line_dep.split(), line_tag.split()):
            res += bpe + sep + lemma + sep + pos + sep + dep + sep + tag + ' '
        res = res[:-1] + '\n'
        if index % 1000 == 0:
            print('Processed', index+1, 'sentences')
    return res



def main():
    for dataset in ['train', 'valid', 'test']:
        print('Loading', dataset, flush=True)
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG), 'r') as file:
            text_bpe = file.read()
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG + '_lemmas'), 'r') as file:
            text_lemma = file.read()
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG + '_pos'), 'r') as file:
            text_pos = file.read()
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG + '_deps'), 'r') as file:
            text_dep = file.read()
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG + '_tags'), 'r') as file:
            text_tag = file.read()
        res = format_opennmt(text_bpe, text_lemma, text_pos, text_dep, text_tag)
        with open(os.path.join(PATH, 'opennmt.' + dataset + '.bpe.' + LANG + '.txt')) as file:
            file.write(res)


if __name__ == "__main__":
    main()
