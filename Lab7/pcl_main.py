##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##

import argparse
import sys
import numpy as np
from collections import Counter

from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_predict

from PCLDataReader import PCLVocab, PCLFeatures, PCLLabels


class BinaryLabels(PCLLabels):
    """For condescension: "true" or "false" """

    def _extract_label(self, example):
        """returns the example xml elements condescension label"""
        return example.get("condescension")

class CategoryLabels(PCLLabels):
    '''For category: i.e. "refugees"'''

    def _extract_label(self, example):
        """returns the examples category """
        return example.attrib['category']

class BagOfWordsFeatures(PCLFeatures):
    """"""
    def __getitem__(self, item):
        """ Returns an interpretable label for the feature at index i """
        return self.vocab.index_to_label(item)

    def _get_num_features(self):
        """ Return the total number of features """

        return self.vocab.__len__()

    def _extract_features(self, example):
        '''should be the counts of words in the example.
        Only include the counts for words that are already stored
        in your input vocabulary vocab.
         Words that are not in the input vocabulary should be ignored'''
        text = self.extract_text(example)
        cnt =  Counter([token for token in text if self.vocab.__getitem__(token) is not None])

        features =[(self.vocab.__getitem__(tuple[0]) ,tuple[1]) for tuple in cnt.most_common()]
        return features

def do_experiment(args):
    # do we need to do this all per - example?

    # raise NotImplementedError
    # Create a PCLVocab object reading in parameters from args
    vocab = PCLVocab(args.vocabulary, args.vocab_size, args.stop_words)

    # create a BagOfWordsFeaturesObject to be utilized later
    bowFeatures = BagOfWordsFeatures(vocab)

    # Binary Labels and category labels is how we label our code.
    # These can be read from the same file, but you may need to call seek(0) from the file pointer.
    binaryLabeler = BinaryLabels()
    categorylabeler = CategoryLabels()

    multinomialNBClassifier = MultinomialNB()

    # create feature and target matrices
    X_train, train_ids = bowFeatures.process(args.training, args.train_size)
    print(X_train.getrow(0))
    print(np.sum(X_train, axis=0))

    #train_labels = binaryLabeler.process(args.labels)[:args.train_size]
    train_labels = binaryLabeler.process(args.labels, args.train_size)
    categoryLabels = categorylabeler.process(args.labels, args.train_size)

    if args.test_category is not None:
        # you will use all of the examples from that category as test data,
        # get the category number that the args is
        category_match = categorylabeler.labels[args.test_category]

        # indexes at which the test_category appears/ not
        test_indices = []
        train_indices = []
        # go through the category labels. can only be as long as the train size?
        for i, category in enumerate(categoryLabels):
        #for i, category in enumerate(categoryLabels[:args.train_size]):
            if category == category_match:
                test_indices.append(i)
            else:
                train_indices.append(i)

        X_train_copy = X_train.copy()
        X_test = X_train_copy[test_indices]

        binary_labels_copy = train_labels.copy()
        labels_test = [train_labels[index] for i, index in enumerate(test_indices)]

        # keep only the examples without that category as training data
        X_train = X_train[train_indices]
        print(test_indices)
        test_ids = [train_ids[i] for i in test_indices]

        print(X_test)

        train_labels = [train_labels[index] for i, index in enumerate(train_indices)]

        # fit your model to then data and labels for examples outside the category test_category
        multinomialNBClassifier = multinomialNBClassifier.fit(X_train, train_labels)

        # get predictions and probabilities for each instance matching the test category
        # you are still doing the same binary classification here
        y_pred = multinomialNBClassifier.predict(X_test)
        class_probs = multinomialNBClassifier.predict_proba(X_test)

    else:
        # if args.xvalidate is not None:
        # perform cross validation on x folds
        y_pred = cross_val_predict(multinomialNBClassifier, X_train, train_labels, cv=args.xvalidate, method='predict')
        class_probs = cross_val_predict(multinomialNBClassifier, X_train, train_labels, cv=args.xvalidate,
                                        method='predict_proba')

    print(y_pred)
    print(sum(y_pred))
    print(class_probs)

    # write to args.output file
    # for each test instance
    for i in range(len(y_pred)):
        # add instance id, predicated class, confidence, to output args
        instance_id = test_ids[i]
        if y_pred[i] == 0:
            pred_class = 'false'
        else:
            pred_class = 'true'
        confidence = class_probs[i][y_pred[i]]

        args.output_file.write(instance_id + " " + pred_class + " " + str(confidence) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("training", type=argparse.FileType('rb'), help="Training instances")
    parser.add_argument("labels", type=argparse.FileType('rb'), help="Training instance labels")
    parser.add_argument("vocabulary", type=argparse.FileType('r'), help="Vocabulary")
    parser.add_argument("-o", "--output_file", type=argparse.FileType('w'), default=sys.stdout, help="Write predictions to FILE", metavar="FILE")
    parser.add_argument("-v", "--vocab_size", type=int, metavar="N", help="Only count the top N words from the vocab file", default=None)
    parser.add_argument("-s", "--stop_words", type=int, metavar="N", help="Exclude the top N words as stop words", default=None)
    parser.add_argument("--train_size", type=int, metavar="N", help="Only train on the first N instances. N=None (default) means use all training instances.", default=None)

    eval_group = parser.add_mutually_exclusive_group(required=True)
    eval_group.add_argument("-t", "--test_category")
    eval_group.add_argument("-x", "--xvalidate", type=int)

    args = parser.parse_args()
    do_experiment(args)

    for fp in (args.output_file, args.training, args.labels, args.vocabulary): fp.close()
