import os
import unicodedata

import sentencepiece as spm

FLORES_DATA_PATH = '/home/usuaris/veu/jordi.armengol/tfg/new/data/flores'
BPE_MODEL_PATH = os.path.join(FLORES_DATA_PATH, 'data-bin', 'wiki_ne_en_bpe5000', 'sentencepiece.bpe.model')
s = spm.SentencePieceProcessor()
s.Load(BPE_MODEL_PATH)

PATH = '/home/usuaris/veu/jordi.armengol/tfg/new/data/flores/data/wiki_ne_en_bpe5000'
LANG = 'en'

def normalize_token(token):
    pieces = s.EncodeAsPieces(token)
    return ''.join(pieces).replace('\u2581', '')


def align_sentencepiece(text_bpe, text_token, text_lemma, text_pos, text_dep, text_tag):
    index_bpe = 0
    repeated_tokens = ''
    repeated_lemmas = ''
    repeated_pos = ''
    repeated_deps = ''
    repeated_tags = ''
    subword_tags = ''
    n_lines = len(text_bpe.splitlines())
    i = 0
    for line_bpe, line_token, line_lemma, line_pos, line_dep, line_tag in zip(text_bpe.splitlines(), text_token.splitlines(), text_lemma.splitlines(), text_pos.splitlines(), text_dep.splitlines(), text_tag.splitlines()):
        index_bpe = 0
        i += 1
        #if i < 80000:#< 61222:
        #    continue
        skip = 0
        for index, token in enumerate(line_token.split()):
            if skip > 0:
                skip -= 1
                continue
            #token = token.replace(chr(8203),'').replace(chr(8206),'').replace('…','...').replace('º','o').replace('™','TM').replace('( ゜o゜ )','( ▁ ゚ o ▁ ゚ )')
            #token = unicodedata.normalize('NFKC', token)
            token = normalize_token(token)
            current_word = ''
            counter = 0
            currently_in_space = False
            while not current_word == token: # and not (token == '…' and current_word == '...') and not (token == 'º' and current_word == 'o') and not(token == '….' and current_word == '....'):
                if index_bpe >= len(line_bpe):
                    print(token)
                    print('current_word', current_word)
                    print('out of bounds', line_token, index_bpe, len(line_bpe)-1)
                    exit()
                if line_bpe[index_bpe] == '\u2581' or line_bpe[index_bpe] == ' ':
                    if line_bpe[index_bpe] == '\u2581' and (index_bpe == 0 or line_bpe[index_bpe-1] == ' ') and (index_bpe == len(line_bpe)-1 or line_bpe[index_bpe+1] == ' '):
                        repeated_tokens += '\u2581' + ' '
                        repeated_lemmas += '\u2581' + ' '
                        repeated_pos += '\u2581' + ' '
                        repeated_deps += '\u2581' + ' '
                        repeated_tags += '\u2581' + ' '
                        subword_tags += 'O' + ' '
                        current_word = ''
                        index_bpe += 1
                        # print('OK')
                        continue
                    index_bpe += 1
                    if not currently_in_space:
                        counter += 1
                    currently_in_space = True
                    ''' <-
                    elif index+2 < len(line_token.split()) and line_lemma.split()[index+1] == '"' and line_lemma.split()[index+2] == '"':
                        print('SOM-HI')
                        repeated_tokens += token + line_token.split()[index+1] + line_token.split()[index+2] + ' '
                        repeated_lemmas += line_lemma.split()[index] + line_lemma.split()[index+1] + line_lemma.split()[index+2] + ' '
                        repeated_pos += line_pos.split()[index] + line_pos.split()[index+1] + line_pos.split()[index+2] + ' '
                        repeated_deps += line_dep.split()[index] + line_dep.split()[index+1] + line_dep.split()[index+2] + ' '
                        repeated_tags += line_tag.split()[index] + line_tag.split()[index+1] + line_tag.split()[index+2] + ' '
                        subword_tags += 'O' + ' '
                        current_word = ''
                        index_bpe += 2
                        skip = 2
                        continue
                    '''
                else:
                    current_word += line_bpe[index_bpe]
                    index_bpe += 1
                    currently_in_space = False
                #current_word = unicodedata.normalize('NFKC', current_word)
                '''
                if 'Anti-traditionalism' in line_token:
                    if 'and' in token:
                        for c in token:
                            print(c, ord(c))
                        #exit()
                    print(index_bpe, token,len(token), len(current_word), current_word)
                '''
            #if index_bpe < len(line_bpe) and line_bpe[index_bpe] not in [' ', '\u2581']: # N features map to 1 subword, eg. " . -> ". FTM, assume only one extra token.
            #if index_bpe < len(line_bpe) and line_bpe[index_bpe] not in [' ', '\u2581'] and index + 1 < len(
            #            line_token.split()) and counter == 1:
            if index_bpe < len(line_bpe) and ((line_bpe[index_bpe] in ['"']) or (line_bpe[index_bpe] in ['s'] and line_bpe[index_bpe-2:index_bpe] == 'it')) and index + 1 < len(
                        line_token.split()) and counter == 1:
                #print('hola')
                #print(ord(text_bpe[index_bpe]))
                #print()
                #until_index = index + 1
                #accum_token = token
                #while accum_token + normalize_token(line_token.split()[until_index] != current_word + text_bpe[index_bpe+1]:
                if counter == 1:
                    # try:
                    repeated_tokens += token + line_token.split()[index+1] + ' '
                    repeated_lemmas += line_lemma.split()[index] + line_lemma.split()[index+1] + ' '
                    repeated_pos += line_pos.split()[index] + line_pos.split()[index+1] + ' '
                    repeated_deps += line_dep.split()[index] + line_dep.split()[index+1] + ' '
                    repeated_tags += line_tag.split()[index] + line_tag.split()[index+1] + ' '
                    subword_tags += 'O' + ' '
                    #except:
                    #    print(line_token)
                    #    exit()
                else:
                    repeated_tokens += token + line_token.split()[index + 1] + ' '
                    repeated_lemmas += line_lemma.split()[index] + line_lemma.split()[index + 1] + ' '
                    repeated_pos += line_pos.split()[index] + line_pos.split()[index + 1] + ' '
                    repeated_deps += line_dep.split()[index] + line_dep.split()[index + 1] + ' '
                    repeated_tags += line_tag.split()[index] + line_tag.split()[index + 1] + ' '
                    subword_tags += 'B' + ' '
                    counter -= 1
                    while counter > 1:
                        repeated_tokens += token + line_token.split()[index + 1] + ' '
                        repeated_lemmas += line_lemma.split()[index] + line_lemma.split()[index + 1] + ' '
                        repeated_pos += line_pos.split()[index] + line_pos.split()[index + 1] + ' '
                        repeated_deps += line_dep.split()[index] + line_dep.split()[index + 1] + ' '
                        repeated_tags += line_tag.split()[index] + line_tag.split()[index + 1] + ' '
                        subword_tags += 'I' + ' '
                        counter -= 1
                    repeated_tokens += token + line_token.split()[index + 1] + ' '
                    repeated_lemmas += line_lemma.split()[index] + line_lemma.split()[index + 1] + ' '
                    repeated_pos += line_pos.split()[index] + line_pos.split()[index + 1] + ' '
                    repeated_deps += line_dep.split()[index] + line_dep.split()[index + 1] + ' '
                    repeated_tags += line_tag.split()[index] + line_tag.split()[index + 1] + ' '
                    subword_tags += 'E' + ' '
                current_word = ''
                skip = 1
                index_bpe += 1
                continue
            if counter == 1:
                repeated_tokens += token + ' '
                repeated_lemmas += line_lemma.split()[index] + ' '
                repeated_pos += line_pos.split()[index] + ' '
                repeated_deps += line_dep.split()[index] + ' '
                repeated_tags += line_tag.split()[index] + ' '
                subword_tags += 'O' + ' '
            else:
                repeated_tokens += token + ' '
                repeated_lemmas += line_lemma.split()[index] + ' '
                repeated_pos += line_pos.split()[index] + ' '
                repeated_deps += line_dep.split()[index] + ' '
                repeated_tags += line_tag.split()[index] + ' '
                subword_tags += 'B' + ' '
                counter -= 1
                while counter > 1:
                    repeated_tokens += token + ' '
                    repeated_lemmas += line_lemma.split()[index] + ' '
                    repeated_pos += line_pos.split()[index] + ' '
                    repeated_deps += line_dep.split()[index] + ' '
                    repeated_tags += line_tag.split()[index] + ' '
                    subword_tags += 'I' + ' '
                    counter -= 1
                repeated_tokens += token + ' '
                repeated_lemmas += line_lemma.split()[index] + ' '
                repeated_pos += line_pos.split()[index] + ' '
                repeated_deps += line_dep.split()[index] + ' '
                repeated_tags += line_tag.split()[index] + ' '
                subword_tags += 'E' + ' '
            current_word = ''
        '''
        try:
            repeated_tokens = repeated_tokens[:-1] + '\n'
            repeated_lemmas = repeated_lemmas[:-1] + '\n'
            repeated_pos = repeated_pos[:-1] + '\n'
            repeated_deps = repeated_deps[:-1] + '\n'
            repeated_tags = repeated_tags[:-1] + '\n'
            subword_tags = subword_tags[:-1] + '\n'
        except:
            repeated_tokens += '\n'
            repeated_lemmas += '\n'
            repeated_pos += '\n'
            repeated_deps += '\n'
            repeated_tags += '\n'
            subword_tags += '\n'
        '''
        repeated_tokens += '\n'
        repeated_lemmas += '\n'
        repeated_pos += '\n'
        repeated_deps += '\n'
        repeated_tags += '\n'
        subword_tags += '\n'
        if len(line_bpe.split()) != len(repeated_lemmas.splitlines()[i-1].split()):
            print(len(line_bpe.split()), 'vs', len(repeated_lemmas.splitlines()[i-1].split()))
            print(line_bpe)
            print(repeated_lemmas.splitlines()[i-1])
            print()
            print('--------------------------------')
            print()
    return repeated_tokens, repeated_lemmas, repeated_pos, repeated_deps, repeated_tags, subword_tags


def main():
    for dataset in ['train', 'valid', 'test']:
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG), 'r') as file:
            text_bpe = file.read()
        with open(os.path.join(PATH, dataset + '.' + LANG + '_tokens'), 'r') as file:
            text_token = file.read()
        with open(os.path.join(PATH, dataset + '.' + LANG + '_lemmas'), 'r') as file:
            text_lemma = file.read()
        with open(os.path.join(PATH, dataset + '.' + LANG + '_pos'), 'r') as file:
            text_pos = file.read()
        with open(os.path.join(PATH, dataset + '.' + LANG + '_deps'), 'r') as file:
            text_dep = file.read()
        with open(os.path.join(PATH, dataset + '.' + LANG + '_tags'), 'r') as file:
            text_tag = file.read()
        repeated_tokens, repeated_lemmas, repeated_pos, repeated_deps, repeated_tags, subword_tags = align_sentencepiece(text_bpe, text_token, text_lemma, text_pos, text_dep, text_tag)
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG + '_tokens'), 'w') as file:
            file.write(repeated_tokens)
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG + '_lemmas'), 'w') as file:
            file.write(repeated_lemmas)
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG + '_pos'), 'w') as file:
            file.write(repeated_pos)
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG + '_deps'), 'w') as file:
            file.write(repeated_deps)
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG + '_tags'), 'w') as file:
            file.write(repeated_tags)
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG + '_subword_tags'), 'w') as file:
            file.write(subword_tags)



if __name__ == "__main__":
    main()
