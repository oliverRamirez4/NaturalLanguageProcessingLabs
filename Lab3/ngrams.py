#!/usr/bin/env python3

import argparse
from lxml import etree
from collections import Counter
from spacy.lang.en import English
import html

nlp = English(pipeline=[], max_length=5000000)


def do_xml_parse(fp, tag):
    """ 
    Iteratively parses XML files
    """
    fp.seek(0)

    for (event, elem) in etree.iterparse(fp, tag=tag):
        yield elem
        elem.clear()

def get_examples(args, attribute, value):
    unigramCounter = Counter()
    bigramCounter = Counter()
    trigramCounter = Counter()
    for elem in do_xml_parse(args.examples, 'example'):
       if elem.attrib[attribute] == value:
            for text in elem.itertext():
                text = html.unescape(text)
                doc = nlp(text)
                unigramCounter += Counter(get_unigrams(doc))
                bigramCounter += Counter(get_bigrams(doc))
                trigramCounter += Counter(get_trigrams(doc))


   
    return (unigramCounter,bigramCounter, trigramCounter)

def get_unigrams(doc, do_lower=True): 
    if do_lower:
        return [x.text.lower() for x in doc]
    else:
        return [x.text for x in doc]

def get_bigrams(doc):
    unigrams = get_unigrams(doc)
    return zip(unigrams[:-1],unigrams[1:])


def get_trigrams(doc):
    unigrams = get_unigrams(doc)
    return zip(unigrams[:-2], unigrams[1:-1], unigrams[2:])

def compare(train, test, unique=False):
    notInTrainSet = 0
    total = 0

    #for types
    if unique == True:
        for token in test.keys():
            if token not in train.keys():
                notInTrainSet += 1
        total = len(test.keys())
    
    #for tokens
    elif unique == False:
        for token in test.elements():
            if token not in train.keys():
                notInTrainSet += 1
        total = sum(test.values())
    return (notInTrainSet,total)

def do_experiment(args, attribute, train_value, test_value): 
    """Print a pandoc-compatible table of experiment results"""
    trainUnigram, trainBigram, trainTrigram = get_examples(args, attribute, train_value) 
    testUnigram, testBigram, testTrigram = get_examples(args, attribute, test_value)

    table_header = "Results for {}, using {} as train and {} as test:"
    print(table_header.format(attribute, train_value, test_value))

    print("| Order | Type/Token | Total | Zeros | % Zeros | ")
    print("| ----  | ---------- | ----- | ----- | ------- | ")
    table_row = "| {gramLevel} | {typetoken} | {total} | {zeros} | {pct:.1%} | "

    
    for do_types in (True, False):
        typetoken = "Type" if do_types else "Token" 
        num_zeros, N = compare(trainUnigram, testUnigram, do_types)
        print(table_row.format( gramLevel = 'Unigram', typetoken=typetoken, 
              total=N, zeros=num_zeros, pct=num_zeros/N))
        
        num_zeros, N = compare(trainBigram, testBigram, do_types)
        print(table_row.format( gramLevel = 'Bigram', typetoken=typetoken, 
              total=N, zeros=num_zeros, pct=num_zeros/N))
        
        num_zeros, N = compare(trainTrigram, testTrigram, do_types)
        print(table_row.format( gramLevel = 'Trigram', typetoken=typetoken, 
              total=N, zeros=num_zeros, pct=num_zeros/N))
    print()

def main(args):
    do_experiment(args, 'randomchunk', 'b', 'a')

    cntTrue = get_examples(args, 'condescension', 'true')
    cntFalse = get_examples(args, 'condescension', 'false')
   

    
    

    '''
    cnt = get_examples(args, 'condescension', 'true')
    print(cnt['the'])
    print(cnt['opportunity'])
    print(cnt['zero'])
    print(cnt['sdfdf'])
    print(cnt.keys())
    '''



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 'rb' means "read as bytes", which means that it doesn't assume
    # the data is UTF-8 text when it's read in.
    parser.add_argument("--examples", "-a",
                        type=argparse.FileType('rb'),
                        help="Content of examples")

    args = parser.parse_args()

    main(args)
