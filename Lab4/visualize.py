#!/usr/bin/env python3

import glove
from sklearn.decomposition import PCA  # put this at the top of your program
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from similarity import *
import argparse
from itertools import chain
import numpy as np
from sklearn.decomposition import PCA  # put this at the top of your program



def read_relations(fp):
    return [tuple(line.split()) for line in fp.readlines()[1:]]

#Must stack words from relations into one matrix first using numpy.vstack(first_vectors, second_vectors)
def perform_pca(array, n_components = 2):
    # For the purposes of this lab, n_components will always be 2.
    pca = PCA(n_components=n_components)
    pc = pca.fit_transform(array)
    return pc

def extract_words(vectors, word_list, relations):
    relationsFiltered = [pair for pair in relations if pair[0] in word_list and pair[1] in word_list]
    word1vectors = [vectors[word_list.index(pair[0])] for pair in relationsFiltered]
    word2vectors = [vectors[word_list.index(pair[1])] for pair in relationsFiltered]

    return word1vectors, word2vectors, relationsFiltered



def plot_relations(pca_first, pca_second, pca_relations, filename='plot.png'):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)


    ax.scatter(pca_first[:, 0], pca_first[:, 1], c='r', s=50)
    ax.scatter(pca_second[:, 0], pca_second[:, 1], c='b', s=50)

    for i in range(len(pca_first)):
        (x, y) = pca_first[i]
        plt.annotate(pca_relations[i][0], xy=(x, y), color="black")
        (x, y) = pca_second[i]
        plt.annotate(pca_relations[i][1], xy=(x, y), color="black")

        (x1, y1) = pca_first[i]
        (x2, y2) = pca_second[i]
        ax.plot((x1, x2), (y1, y2), linewidth=1, color="lightgray")

    plt.savefig(filename)
        
    

def main(args):
    vectors, word_list = glove.load_glove_vectors(args.npyFILE)
    relations = read_relations(args.relationsFILE)
    print(relations)


    # Filter through the relations, get vectors for each word, but not for relations that do not exist
    word1vectors, word2vectors, relationsFiltered = extract_words(vectors, word_list, relations)

    #put the two vectors int one array, filter back into two vectors after pca
    array = np.vstack((word1vectors, word2vectors))
    pca_vectors = perform_pca(array, 2)
    pca_first = pca_vectors[:len(relationsFiltered)]
    pca_second = pca_vectors[len(relationsFiltered):]

    #plot the graph
    plot_relations(pca_first, pca_second, relationsFiltered)





    
    # ...

    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the n closest words to " +\
                      "a given word (if specified) or to all of the " +\
                       "words in a text file (if specified). If " +\
                       "neither is specified, compute nothing.")
    parser.add_argument("npyFILE",
                        type=argparse.FileType('rb'),
                        help='an .npy file to read the saved numpy data from')
    parser.add_argument("relationsFILE",
                        type=argparse.FileType('r'),
                        help='a file containing pairs of relations')
    parser.add_argument("--plot", "-p", default="plot.png", help="Name of file to write plot to.")

    args = parser.parse_args()
    main(args)
