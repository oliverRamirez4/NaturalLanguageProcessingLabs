#!/usr/bin/env python3
import re
from collections import Counter
import os

def get_words(s, do_Lower = False): 
    if do_Lower:
        s = s.lower()
    return s.split()

def count_words(list_of_words): 
    cnt = Counter()
    for word in list_of_words:
        cnt[word] += 1
    return cnt


def words_by_frequency(list_of_words, n=None):
    cnt = count_words(list_of_words)
    return cnt.most_common(n)

def tokenize(s, do_lower=False):

    if do_lower:
        s = s.lower()

    tokens = re.findall(r'\w+|[.,;!?\'"-|{}]', s)
    
    return tokens


def filter_nonwords(list_of_tokens):
    filtered = []
    for token in list_of_tokens:
        if token.isalpha():
            filtered.append(token)
    return filtered

def main(): 
    directory = "/cs/cs159/data/gutenberg/"
    files = os.listdir(directory)
    for file in files:
        infile = open(os.path.join(directory, file), 'r', encoding='latin1')
        s = infile.read()
        list_of_tokens =tokenize(s, False)
        wordsOnly = filter_nonwords(list_of_tokens)
        print(words_by_frequency(wordsOnly, 5))


    

            
if __name__ == '__main__':
    main()
