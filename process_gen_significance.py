PATH = ''


def get_ref_gen(orig):
    ref = ''
    gen = ''
    for l in orig:
        if l[0] == 'T':
            ref += ' '.join(l.split()[2:]) + '\n'
        elif l[0] == 'H':
            gen += ' '.join(l.split()[2:]) + '\n'
    return ref, gen


def main():
    with open(PATH, 'r') as fo:
        orig = fo.readlines()
    ref, gen = get_ref_gen(orig)
    with open(PATH + '.ref') as fr:
        fr.write(ref)
    with open(PATH + '.gen') as fr:
        fr.write(gen)


if __name__ == '__main__':
    main()
