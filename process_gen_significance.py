PATH = ''


def reverse_splitlines(lines):
    s = ''
    for l in lines:
        s += l + '\n'
    return s


def get_ref_gen(orig):
    n_sentences = int(orig[-2].split()[2])
    ref = [None]*n_sentences
    gen = [None]*n_sentences
    for l in orig:
        if l[0] == 'T':
            ref[int(l.split()[0][2:])] = ' '.join(l.split()[2:])
        elif l[0] == 'H':
            gen[int(l.split()[0][2:])] = ' '.join(l.split()[2:])
    ref = reverse_splitlines(ref)
    gen = reverse_splitlines(gen)
    return ref, gen


def main():
    with open(PATH, 'r') as fo:
        orig = fo.readlines()
    ref, gen = get_ref_gen(orig)
    with open(PATH + '.ref', 'w') as fr:
        fr.write(ref)
    with open(PATH + '.gen', 'w') as fr:
        fr.write(gen)


if __name__ == '__main__':
    main()
