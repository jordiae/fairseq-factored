#import stanfordnlp
#from spacy_stanfordnlp import StanfordNLPLanguage
import os
import sys

PATH = '/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14.tokenized.de-en/tmp'
LANG = 'de'

#snlp = stanfordnlp.Pipeline(lang=LANG, disable=['ner'])
#nlp = StanfordNLPLanguage(snlp)

import spacy
nlp = spacy.load("de_core_news_sm")

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
        if index_line % 100 == 0:
            print('Processed', index_line, 'sentences', flush=True)
            #global snlp
            #global nlp
            #del snlp
            #del nlp
            #snlp = stanfordnlp.Pipeline(lang=LANG, disable=['ner'])
            #nlp = StanfordNLPLanguage(snlp)
            #nlp = spacy.load("de_core_news_sm")
    return text_token, text_lemma, text_pos, text_dep, text_tag

def is_null(x):
    if x is None or x == '':
        return True
    return False
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
        if is_null(token.text) or is_null(token.lemma_) or is_null(token.pos_) or is_null(token.dep_) or is_null(token.tag):
            raise Exception('NULL TAG!')
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
        for (index, (token, lemma, pos, dep, tag)) in enumerate(zip(tokens, lemmas, poss, deps, tags)):
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
            #del snlp
            #del nlp
            #snlp = stanfordnlp.Pipeline(lang=LANG, disable=['ner'])
            #nlp = StanfordNLPLanguage(snlp)
    return text_token, text_lemma, text_pos, text_dep, text_tag


def main():
    for dataset in ['train', 'valid', 'test']:
        print('Loaded', os.path.join(PATH, dataset + '.' + LANG), flush=True)
        with open(os.path.join(PATH, dataset + '.' + LANG), 'r') as file:
            text = file.read()
        print("Running tagger...", flush=True)
        text_token, text_lemma, text_pos, text_dep, text_tag = tag_text(text)  # tag_text(text)
        return
        with open(os.path.join(PATH, dataset + '.' + LANG + '_tokens'), 'w', encoding="utf8") as file:
            file.write(text_token)
        with open(os.path.join(PATH, dataset + '.' + LANG + '_lemmas'), 'w', encoding="utf8") as file:
            file.write(text_lemma)
        with open(os.path.join(PATH, dataset + '.' + LANG + '_pos'), 'w', encoding="utf8") as file:
            file.write(text_pos)
        with open(os.path.join(PATH, dataset + '.' + LANG + '_deps'), 'w', encoding="utf8") as file:
            file.write(text_dep)
        with open(os.path.join(PATH, dataset + '.' + LANG + '_tags'), 'w', encoding="utf8") as file:
            file.write(text_tag)


if __name__ == "__main__":
    #sys.settrace(main())
    main()
