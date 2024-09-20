#!/usr/bin/env python3

from glove import *
from similarity import *
from visualize import *
import argparse
import numpy as np
import random
import sys

def average_difference(first_vectors, second_vectors):

    vec_list = []
    for i in range(len(first_vectors)):
        vec = []
        for k in range(len(first_vectors[0])):
            vec.append(first_vectors[i][k]-second_vectors[i][k])
        vec_list.append(vec)
    vec_array= numpy.array(vec_list)
    return numpy.divide(sum(vec_array),len(vec_array))

def do_experiment(args):
    print('File name:', args.relationsFILE.name)

    # setup
    relations = read_relations(args.relationsFILE)
    random.shuffle(relations)
    vectors, word_list = load_glove_vectors(args.npyFILE)

    # clean  the relations of pairs not in wordlist, don't need to call
    #the whole extract_words function for this
    relationsFiltered = [pair for pair in relations if pair[0] in word_list and pair[1] in word_list]

    # divide the filtered relations into a training and a testing set
    training_relations = relationsFiltered[:int(len(relationsFiltered) * 0.8)]
    test_relations = relationsFiltered[int(len(relationsFiltered) * 0.8):]

    #get the vectors for each of the corresponding words in both train set and test set
    firstVectorsTrain, secondVectorsTrain, training_relations = extract_words(vectors, word_list, training_relations)
    firstVectorsTest, secondVectorsTest, test_relations = extract_words(vectors, word_list, test_relations)

    #step 4) Using your average_difference function, find the average difference between all of the words in your training_relations.
    avgDistanceVector = average_difference(firstVectorsTrain, secondVectorsTrain)


    # get the 100 closest words in secword100closest
    #get the 100 closest words to predicted related words in closest100PredictedVec
    # to each second word at index i of test_relations
    secWord100Closest = []
    closest100PredictedVec = []
    # get the 100 most similar words for each of the second word in test_relations
    #while removing the first word for each one, hence n=101

    for secondVector in secondVectorsTest:
        secWord100Closest.append([tuple[1] for tuple in closest_vectors(secondVector, word_list, vectors, 101)[1:]])
        currentVector = (secondVector + avgDistanceVector)
        closest100PredictedVec.append([tuple[1] for tuple in closest_vectors(currentVector, word_list, vectors, 100)])
    # question 1(1): How often is the first word in the original relation pair also the most similar word?
    # question 1(2): How often is the first word in the relation in the top 10 most similar words?
    # question 1(3): Report the mean reciprocal rank for where you found the first word in the results.
    top_word_count = 0
    top_ten_count = 0
    mean_reciprocal_sum = 0.0
    for index in range(len(test_relations)):
        firstWord = test_relations[index][0]
        if firstWord == secWord100Closest[index][0]:
            top_word_count += 1
        if firstWord in secWord100Closest[index][:10]:
            top_ten_count += 1
        if firstWord in secWord100Closest[index]:
            mean_reciprocal_sum += 1/(secWord100Closest[index].index(firstWord)+1)
    mrr = mean_reciprocal_sum/len(test_relations)
    print('Question 1: first word relatedness to closest words of second word')
    print("the first word in the relation pair is the most similar to the second word ",top_word_count, " out of ", len(test_relations), " times. Or ", round(top_word_count/len(test_relations)*100, 2), "%")
    print("The first word in the relation pair is in the top 10 most similar words to the second word ",top_ten_count, " out of ", len(test_relations), " times. Or ", round(top_ten_count/len(test_relations)*100, 2), "%")
    print("The mean reciprocal rank is ", round(mrr, 4))


    #question2

    top_word_count = 0
    top_ten_count = 0
    mean_reciprocal_sum = 0.0
    for index in range(len(test_relations)):
        firstWord = test_relations[index][0]
        if firstWord == closest100PredictedVec[index][0]:
            top_word_count += 1
        if firstWord in closest100PredictedVec[index][:10]:
            top_ten_count += 1
        if firstWord in closest100PredictedVec[index]:
            mean_reciprocal_sum += 1 / (closest100PredictedVec[index].index(firstWord) + 1)
    mrr = mean_reciprocal_sum / len(test_relations)
    print("Question 2: Using the predicted first vector along with average distance")
    print("the first word in the relation pair is the most similar to the second_word_vector + avg_distance_vector ", top_word_count, " out of ",
          len(test_relations), " times. Or ", round(top_word_count/len(relations)*100, 2), "%")
    print("The first word in the relation pair is in the top 10 most similar words to the second_word_vector + avg_distance_vector ", top_ten_count,
          " out of ", len(test_relations), " times. Or ", round(top_ten_count/len(test_relations)*100, 2), "%")
    print("The mean reciprocal rank is ", round(mrr, 4))

def main(args):
    do_experiment(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npyFILE",
                        type=argparse.FileType('rb'),
                        help='an .npy file to read in the array and words from')
    parser.add_argument("relationsFILE",
                        type=argparse.FileType('r'),
                        help="a file containing pairs of relations")
    args = parser.parse_args()
    main(args)

