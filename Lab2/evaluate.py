#!/usr/bin/env python3

import argparse, os
from tokenizer import tokenize

def concordance(words, i, eval=''):
    concordance = {
        'prechars': ' '.join([x for x in words[max(0,i-20):i]])[-30:],
        'center': words[i],
        'postchars': ' '.join([x for x in words[i+1:]])[:30],
        'label': '%s %6d' % (eval, i)
    }

    print("{prechars: >30}  {center: <20}  {postchars: <30}  ({label})".format(**concordance))    

def evaluate(tokens, reference_file, hypothesis_file, verbose=0):
    reference = set([int(x.rstrip()) for x in reference_file])
    hypothesis = set([int(x.rstrip()) for x in hypothesis_file])
    all_tokens = set(range(len(tokens)))

    # Compute confusion matrix
    true_positives = hypothesis & reference
    true_negatives = all_tokens - (hypothesis|reference)
    false_positives = hypothesis - reference
    false_negatives = reference - hypothesis


    # Compute precision, recall, F1
    precision = len(true_positives) / len(hypothesis)
    recall = len(true_positives) / len(reference)
    f = 2*precision*recall/(precision+recall)

    if verbose >= 1:
        words = tokens
        if verbose == 2:
            for i in true_positives: concordance(words, i, 'TP')
            for i in true_negatives: concordance(words, i, 'TN')
        else:
            for i in false_positives: concordance(words, i, 'FP')
            for i in false_negatives: concordance(words, i, 'FN')

    print("TP: {:7d}".format(len(true_positives)), end="")
    print("\tFN: {:7d}".format(len(false_negatives)))

    print("FP: {:7d}".format(len(false_positives)), end="")
    print("\tTN: {:7d}".format(len(true_negatives)))

    print()

    print("PRECISION: {:5.2%}".format(precision), end="")
    print("\tRECALL: {:5.2%}".format(recall), end="")
    print("\tF: {:5.2%}".format(f))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbosity", 
                        type=int, 
                        choices=[0, 1, 2],
                        help="increase output verbosity:\n\t0: show results summary;\n\t" +\
                             "1: show concordance for FP, FN only;\n\t" +\
                             "2: show concordance for TP, FP, FN only",
                        default=0
                        )
    parser.add_argument("-d", "--data_location", help="Path to data directory", default="/data/brown/")
    parser.add_argument("-c", "--category", help="Category of data to use", required=True)
    parser.add_argument("-y", "--hypothesis", help="User-generated file with boundaries to compare to reference", type=argparse.FileType('r'), required=True)

    args=parser.parse_args()

    text_path = os.path.join(args.data_location, "{}.txt".format(args.category))
    reference_path = os.path.join(args.data_location, "{}-eos.txt".format(args.category))

    text = open(text_path).read()
    tokens = tokenize(text)

    evaluate(tokens, open(reference_path), args.hypothesis, args.verbosity)
