#!/usr/bin/env python3

import glove
import numpy
import argparse

def compute_length(a):

    return numpy.linalg.norm(a, axis=a.ndim-1)


def cosine_similarity(array1, array2):
    ## Hint: array2 should be passed into dot first, since we're assuming that array2 will be the (potentially) multidimensional array
    ## Like this:  dot_product = numpy.dot(array2, array1)
    return numpy.divide(numpy.dot(array2, array1), compute_length(array1)*compute_length(array2))

#returns a list of tuples containing the n closest words and their vectors in words_list
def closest_vectors(v, words, array, n):
    similarities = cosine_similarity(v, array)
    to_sort = list(zip(similarities, words))

    return sorted(to_sort, reverse = True )[:n]


def main(args):


    vectors, word_list = glove.load_glove_vectors(args.npyFILE)

    # tests for part 2c
    print(word_list.index('cat'))
    dog_vec = glove.get_vec('dog', word_list, vectors)
    print(compute_length(dog_vec))
    cat_vec = glove.get_vec('cat', word_list, vectors)
    print(cosine_similarity(dog_vec, cat_vec))
    
    # test for part 2d
    closest_vectors(cat_vec, word_list, vectors, 3)
    print(closest_vectors(cat_vec, word_list, vectors, 3))
    
    
    if args.word:
        words = [args.word]
    elif args.file:
        words = [x.strip() for x in args.file]
    else: return

    for word in words:
        print("The {} closest words to {} are:".format(args.num, word))
        word_vector = glove.get_vec(word, word_list, vectors)
        closest = closest_vectors(word_vector, word_list, vectors, args.num)
        for similarity, similar_word in closest:
            print("    * {} (similarity {})".format(similar_word, similarity))
        print()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the n closest words to " +\
                      "a given word (if specified) or to all of the " +\
                       "words in a text file (if specified). If " +\
                       "neither is specified, compute nothing.")
    parser.add_argument("--word", "-w", metavar="WORD", help="a single word")
    parser.add_argument("--file", "-f", metavar="FILE", type=argparse.FileType('r'),
                        help="a text file with one word per line.")
    parser.add_argument("--num", "-n", type=int, default=5,
                        help="find the top n most similar words")
    parser.add_argument("npyFILE",
                        type=argparse.FileType('rb'),
                        help='an .npy file to read the saved numpy data from')

    args = parser.parse_args()
    main(args)
    
