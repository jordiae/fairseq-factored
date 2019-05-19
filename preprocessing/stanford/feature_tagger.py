import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage
import os

PATH = '/home/usuaris/veu/jordi.armengol/tfg/new/data/flores/data/wiki_ne_en_bpe5000'
LANG = 'en'

snlp = stanfordnlp.Pipeline(lang=LANG)
nlp = StanfordNLPLanguage(snlp)


def tag_sentence(sentence):
    doc = nlp(sentence)
    tokens = []
    lemmas = []
    poss = []
    deps = []
    tags = []
    for token in doc:
        tokens.append(token.text)
        lemmas.append(token.lemma_)
        poss.append(token.pos_)
        deps.append(token.dep_)
        tags.append(token.tag_)
    return tokens, lemmas, poss, deps, tags


def tag_text(text):
    lines = text.splitlines()
    text_token = ''
    text_lemma = ''
    text_pos = ''
    text_dep = ''
    text_tag = ''
    for index_line, line in enumerate(lines, start=1):
        tokens, lemmas, poss, deps, tags = tag_sentence(line)
        for (index, (token, lemma, pos, dep, tag)) in enumerate(zip(tokens, lemmas, poss, deps, tags)):
            if index == len(tokens) - 1:
                sep = '\n'
            else:
                sep = ' '
            text_token += token + sep
            text_lemma += lemma + sep
            text_pos += pos + sep
            text_dep += dep + sep
            text_tag += tag + sep
        if index_line % 1000 == 0:
            print('Processed', index_line, 'sentences', flush=True)
    return text_token, text_lemma, text_pos, text_dep, text_tag


def tag_doc(doc):
    tokens = []
    lemmas = []
    poss = []
    deps = []
    tags = []
    for token in doc:
        tokens.append(token.text)
        lemmas.append(token.lemma_)
        poss.append(token.pos_)
        deps.append(token.dep_)
        tags.append(token.tag_)
    return tokens, lemmas, poss, deps, tags


def efficient_tag_text(text):
    lines = text.splitlines()
    docs = nlp.pipe(lines)
    text_token = ''
    text_lemma = ''
    text_pos = ''
    text_dep = ''
    text_tag = ''
    for index_line, doc in enumerate(docs, start=1):
        tokens, lemmas, poss, deps, tags = tag_doc(doc)
        for (index, (token, lemma, pos, dep, tag)) in enumerate(zip(lemmas, poss, deps, tags)):
            if index == len(lemmas) - 1:
                sep = '\n'
            else:
                sep = ' '
            text_token += token + sep
            text_lemma += lemma + sep
            text_pos += pos + sep
            text_dep += dep + sep
            text_tag += tag + sep
        if index_line % 1000 == 0:
            print('Processed', index_line, 'sentences', flush=True)
    return text_token, text_lemma, text_pos, text_dep, text_tag


def main():
    for dataset in ['train', 'valid', 'test']:
        print('Loaded', os.path.join(PATH, dataset + '.' + LANG), flush=True)
        with open(os.path.join(PATH, dataset + '.' + LANG), 'r') as file:
            text = file.read()
        text_token, text_lemma, text_pos, text_dep, text_tag = efficient_tag_text(text)  # tag_text(text)
        with open(os.path.join(PATH, dataset + '.' + LANG + '_tokens'), 'w') as file:
            file.write(text_token)
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