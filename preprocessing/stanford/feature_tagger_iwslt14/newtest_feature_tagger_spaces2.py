#import stanfordnlp
#from spacy_stanfordnlp import StanfordNLPLanguage
import os
import sys

PATH = '/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt13.tokenized.de-en/tmp13'
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
        tokens.append(token.text.replace(' ', '-'))
        lemmas.append(token.lemma_.replace(' ', '-'))
        poss.append(token.pos_.replace(' ', '-'))
        deps.append(token.dep_.replace(' ', '-'))
        tags.append(token.tag_.replace(' ', '-'))
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
            if 'Ã¤' in token:
                raise Exception('Bad umlaut!')
        if index_line % 100 == 0:
            print('Processed', index_line, 'sentences', flush=True)
    return text_token, text_lemma, text_pos, text_dep, text_tag


def main():
    for dataset in ['test']:#['train', 'valid', 'test']:
        print('Loaded', os.path.join(PATH, dataset + '.' + LANG), flush=True)
        with open(os.path.join(PATH, dataset + '.' + LANG), 'r', encoding="utf8") as file:
            text = file.read()
        print("Running tagger...", flush=True)
        text_token, text_lemma, text_pos, text_dep, text_tag = tag_text(text)
        with open(os.path.join(PATH, dataset + '.' + LANG + '_tokensS'), 'w', encoding="utf8") as file:
            file.write(text_token)
        with open(os.path.join(PATH, dataset + '.' + LANG + '_lemmasS'), 'w', encoding="utf8") as file:
            file.write(text_lemma)
        '''
        with open(os.path.join(PATH, dataset + '.' + LANG + '_posS'), 'w', encoding="utf8") as file:
            file.write(text_pos)
        with open(os.path.join(PATH, dataset + '.' + LANG + '_depsS'), 'w', encoding="utf8") as file:
            file.write(text_dep)
        with open(os.path.join(PATH, dataset + '.' + LANG + '_tagsS'), 'w', encoding="utf8") as file:
            file.write(text_tag)
        '''


if __name__ == "__main__":
    main()
