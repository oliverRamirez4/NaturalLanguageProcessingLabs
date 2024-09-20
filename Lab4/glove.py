#!/usr/bin/env python3

import argparse
import numpy


def load_text_vectors(fp):
    columns = len(fp.readline().split()) - 1
    fp.seek(0)
    rows = len(fp.readlines())
    vectors = numpy.zeros((rows, columns))
    fp.seek(0)

    list_of_words = []
    i = 0
    for line in fp.readlines():
        split = line.split()
        list_of_words.append(split[0])
        vector = numpy.array(split[1:]).astype(numpy.float)
        vectors[i] = vector
        i += 1

    return (list_of_words, vectors)


def save_glove_vectors(word_list, vectors, fp):
    numpy.save(fp, vectors)
    numpy.save(fp, word_list)
    fp.close()


def load_glove_vectors(fp):
    array = numpy.load(fp, allow_pickle=True)
    words = list(numpy.load(fp, allow_pickle=True))
    return (array, words)


def get_vec(word, word_list, vectors):
    return vectors[word_list.index(word)]


def main(args):
    words, vectors = load_text_vectors(args.GloVeFILE)
    save_glove_vectors(words, vectors, args.npyFILE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("GloVeFILE",
                        type=argparse.FileType('r'),
                        help="a GloVe text file to read from")
    parser.add_argument("npyFILE",
                        type=argparse.FileType('wb'),
                        help='an .npy file to write the saved numpy data to')

    args = parser.parse_args()
    main(args)
