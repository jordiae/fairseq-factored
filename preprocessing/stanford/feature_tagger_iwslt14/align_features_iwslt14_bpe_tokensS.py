import os
import itertools

PATH = '/home/usuaris/veu/jordi.armengol/tfg/new/data/iwslt14.tokenized.de-en.stanford/tmp'
LANG = 'de_tokensS'

'''
def align_bpe2(text_bpe, text_token, text_lemma, text_pos, text_dep, text_tag):
    repeated_tokens = ''
    repeated_lemmas = ''
    repeated_pos = ''
    repeated_deps = ''
    repeated_tags = ''
    subword_tags = ''
    i = 0
    for line_bpe, line_token, line_lemma, line_pos, line_dep, line_tag in zip(text_bpe.splitlines(), text_token.splitlines(), text_lemma.splitlines(), text_pos.splitlines(), text_dep.splitlines(), text_tag.splitlines()):
        index_bpe = 0
        i += 1
        for index, token in enumerate(line_token.split()):
            current_word = ''
            counter = 0
            currently_in_space = False
            while not current_word == token:
                if index_bpe >= len(line_bpe):
                    print('target token', token)
                    print('current_word', current_word)
                    print('out of bounds: line_token', line_token, index_bpe, len(line_bpe)-1)
                    print('line_bpe', line_bpe)
                    exit()
                if (line_bpe[index_bpe] == '@' and line_bpe[index_bpe+1] == '@') or line_bpe[index_bpe] == ' ':
                    if line_bpe[index_bpe + 1] == '@':
                        index_bpe += 2
                    else:
                        index_bpe += 1
                    if not currently_in_space:
                        counter += 1
                    currently_in_space = True
                else:
                    current_word += line_bpe[index_bpe]
                    index_bpe += 1
                    currently_in_space = False
            if counter == 1:
                repeated_tokens += token + ' '
                repeated_lemmas += line_lemma[index] + ' '
                repeated_pos += line_pos[index] + ' '
                repeated_deps += line_dep[index] + ' '
                repeated_tags += line_tag[index] + ' '
                subword_tags += 'O' + ' '
            else:
                repeated_tokens += token + ' '
                repeated_lemmas += line_lemma[index] + ' '
                repeated_pos += line_pos[index] + ' '
                repeated_deps += line_dep[index] + ' '
                repeated_tags += line_tag[index] + ' '
                subword_tags += 'B' + ' '
                counter -= 1
                while counter > 1:
                    repeated_tokens += token + ' '
                    repeated_lemmas += line_lemma[index] + ' '
                    repeated_pos += line_pos[index] + ' '
                    repeated_deps += line_dep[index] + ' '
                    repeated_tags += line_tag[index] + ' '
                    subword_tags += 'I' + ' '
                    counter -= 1
                repeated_tokens += token + ' '
                repeated_lemmas += line_lemma[index] + ' '
                repeated_pos += line_pos[index] + ' '
                repeated_deps += line_dep[index] + ' '
                repeated_tags += line_tag[index] + ' '
                subword_tags += 'E' + ' '
            current_word = ''
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
    return repeated_tokens, repeated_lemmas, repeated_pos, repeated_deps, repeated_tags, subword_tags


def align_bpe(text_bpe, tags):
    token_index = 0
    tag_index = 0
    splitted_text_bpe = text_bpe.split()
    splitted_tags = tags.split()
    aligned_tags = ''
    end_of_lines_positions = list(itertools.accumulate(list(map(lambda x: len(x.split()), text_bpe.splitlines()))))
    end_of_line_index = 0
    while token_index < len(splitted_text_bpe):
        if '@@' in splitted_text_bpe[token_index]:
            while '@@' in splitted_text_bpe[token_index]:
                #print(splitted_text_bpe[token_index],splitted_tags[tag_index])
                aligned_tags += (splitted_tags[tag_index])
                #aligned_tags += '@@'
                aligned_tags += ' '
                token_index += 1
                if '@@' not in splitted_text_bpe[token_index]:
                    #print(splitted_text_bpe[token_index],splitted_tags[tag_index])
                    aligned_tags += (splitted_tags[tag_index])
                    if token_index == end_of_lines_positions[end_of_line_index] - 1:
                        aligned_tags += '\n'
                        end_of_line_index += 1
                    else:
                        aligned_tags += ' '
                    token_index += 1
                    tag_index += 1
        else:
            #print(splitted_text_bpe[token_index],splitted_tags[tag_index])
            aligned_tags += (splitted_tags[tag_index])
            if token_index == end_of_lines_positions[end_of_line_index] - 1:
                aligned_tags += '\n'
                end_of_line_index += 1
            else:
                aligned_tags += ' '
            token_index += 1
            tag_index += 1
    return aligned_tags
'''

def align_bpe(text_bpe, tags):
    res = ''
    for (index_line, (line_bpe, line_tags)) in enumerate(zip(text_bpe.splitlines(), tags.splitlines())):
        bpe_tokens = line_bpe.split()
        tag_tokens = line_tags.split()
        bpe_index = 0
        tag_index = 0
        while bpe_index < len(bpe_tokens):
            if '@@' in bpe_tokens[bpe_index]:
                while '@@' in bpe_tokens[bpe_index]:
                    res += tag_tokens[tag_index] + ' '
                    bpe_index += 1
                    if '@@' not in bpe_tokens[bpe_index]:
                        res += tag_tokens[tag_index] + ' '
                        bpe_index += 1
                        tag_index += 1
            else:
                res += tag_tokens[tag_index] + ' '
                bpe_index += 1
                tag_index += 1
        if tag_index != len(tag_tokens):
            raise Exception('Ignored tags in line ' + str(index_line))
        res += '\n'
    return res


def main():
    for dataset in ['train', 'valid', 'test']:
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG), 'r') as file:
            text_bpe_tokens = file.read()
        '''
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
        '''
        with open(os.path.join(PATH, dataset + '.' + 'de' + '_lemmas'), 'r') as file:
            text_lemma = file.read()
        with open(os.path.join(PATH, dataset + '.' + 'de' + '_pos'), 'r') as file:
            text_pos = file.read()
        with open(os.path.join(PATH, dataset + '.' + 'de' + '_deps'), 'r') as file:
            text_dep = file.read()
        with open(os.path.join(PATH, dataset + '.' + 'de' + '_tags'), 'r') as file:
            text_tag = file.read()
        #repeated_tokens, repeated_lemmas, repeated_pos, repeated_deps, repeated_tags, subword_tags = align_bpe2(
        #    text_bpe, text_token, text_lemma, text_pos, text_dep, text_tag)
        #repeated_tokens = align_bpe(text_bpe, text_token)
        repeated_lemmas = align_bpe(text_bpe_tokens, text_lemma)
        repeated_pos = align_bpe(text_bpe_tokens, text_pos)
        repeated_deps = align_bpe(text_bpe_tokens, text_dep)
        repeated_tags = align_bpe(text_bpe_tokens, text_tag)
        #with open(os.path.join(PATH, dataset + '.bpe.' + LANG + '_tokens'), 'w') as file:
        #    file.write(repeated_tokens)
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG + '_lemmas'), 'w') as file:
            file.write(repeated_lemmas)
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG + '_pos'), 'w') as file:
            file.write(repeated_pos)
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG + '_deps'), 'w') as file:
            file.write(repeated_deps)
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG + '_tags'), 'w') as file:
            file.write(repeated_tags)
        '''
        with open(os.path.join(PATH, dataset + '.bpe.' + LANG + '_subword_tags'), 'w') as file:
            file.write(subword_tags)
        '''



if __name__ == "__main__":
    main()
