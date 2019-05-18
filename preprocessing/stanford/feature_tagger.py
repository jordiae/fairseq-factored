import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
import os

PATH = '/home/usuaris/veu/jordi.armengol/tfg/new/data/flores/data/wiki_ne_en_bpe5000'
LANG = 'en'

snlp = stanfordnlp.Pipeline(lang=LANG)
nlp = StanfordNLPLanguage(snlp)


def tag_sentence(sentence):
    doc = nlp(sentence)
    lemmas = []
    poss = []
    deps = []
    tags = []
    for token in doc:
        lemmas.append(token.lemma_)
        poss.append(token.pos_)
        deps.append(token.dep_)
        tags.append(token.tag_)
    return lemmas, poss, deps, tags


def tag_text(text):
    lines = text.splitlines()
    text_lemma = ''
    text_pos = ''
    text_dep = ''
    text_tag = ''
    for index, line in enumerate(lines, start=1):
        lemmas, poss, deps, tags = tag_sentence(line)
        for (index, (lemma, pos, dep, tag)) in enumerate(zip(lemmas, poss, deps, tags)):
            if index == len(lemmas) - 1:
                sep = '\n'
            else:
                sep = ' '
            text_lemma += lemma + sep
            text_pos += pos + sep
            text_dep += dep + sep
            text_tag += tag + sep
        if index % 10000:
            print('Processed', index, 'sentences', flush=True)
    return text_lemma, text_pos, text_dep, text_tag


def main():
    for dataset in ['train', 'valid', 'test']:
        with open(os.path.join(PATH, dataset + '.' + LANG), 'r') as file:
            text = file.read()
        text_lemma, text_pos, text_dep, text_tag = tag_text(text)
        with open(os.path.join(PATH, dataset + '.' + LANG + '_lemmas'), 'w') as file:
            file.write(text_lemma)
        with open(os.path.join(PATH, dataset + '.' + LANG + '_pos'), 'w') as file:
            file.write(text_pos)
        with open(os.path.join(PATH, dataset + '.' + LANG + '_deps'), 'w') as file:
            file.write(text_dep)
        with open(os.path.join(PATH, dataset + '.' + LANG + '_tags'), 'w') as file:
            file.write(text_tag)


if __name__ == "__main__":
    main()
