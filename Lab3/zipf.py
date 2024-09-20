#!/usr/bin/env python3

import math
import os.path
from collections import Counter
from spacy.lang.en import English
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import random




nlp = English(pipeline=[], max_length=5000000)


def H_approx(n):
    """
    Returns an approximate value of n-th harmonic number.
    http://en.wikipedia.org/wiki/Harmonic_number
    """
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + math.log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)

def do_zipf_plot(counts, label=""):
    fig = pyplot.figure()
    countsList = counts.most_common()
    total = sum(counts.values())
    types = len(counts)
    frequency = []
    rankList = []
    zipfLawList = []

    for rank in range(len(countsList)):
        rankList.append(rank + 1)
        frequency.append(countsList[rank][1]/total)
        #zipfLawList.append((total/(H_approx(rank + 1) * rank + 1)))
        zipfLawList.append(1/(H_approx(types)*(rank+1)))

    pyplot.loglog(rankList, frequency, linestyle='-', label = 'empirical relative frequency', color = 'b')
    pyplot.loglog(rankList, zipfLawList, linestyle='-', label = 'Zipfs law values', color = 'r')

    pyplot.xlabel('log(rank)')
    pyplot.ylabel('log(relative freq)')
    pyplot.title('Zipfs Law Test for ' + label)
    pyplot.legend(loc='lower left')

    pyplot.savefig('zipf_{}.png'.format(label))

    pyplot.close()
    





def read_all(directory, extension=None):
    cnt = Counter()
    for dirPath, dirNames, fileNames in os.walk(directory):
        if extension is not None:
            fileNames = [filename for filename in fileNames if os.path.splitext(filename)[1] == extension]
    for file in fileNames:
        cnt += read_one(os.path.join(directory, file))
    return cnt


def read_one(fname):
    with open(fname, 'r', encoding='latin1') as fp: 
        doc = nlp(fp.read())
    cnt = Counter([token.text.lower() for token in doc])
    return cnt
        
   


    # do processing...
def plot_all(directory):
    counts = read_all(directory, ".txt")
    do_zipf_plot(counts, os.path.basename(directory))

def plot_one(fname):
    counts = read_one(fname)
    title = os.path.splitext(os.path.basename(fname))[0]

    do_zipf_plot(counts, label=title)

def main():
    #plot_one('/cs/cs159/data/gutenberg/burgess-busterbrown.txt')
    plot_all('/cs/cs159/data/gutenberg')

    #tokens = read_one('/cs/cs159/data/gutenberg/carroll-alice.txt')
  
    '''
    s= ''
    for i in range(2000000):
        s+= random.choice("abcdefg ")

    with open("randomText.txt", 'w') as fp:
        fp.write(s)

    plot_one('randomText.txt')
    '''


if __name__ == "__main__":
    main()
