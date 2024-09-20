#!/usr/bin/env python3

import sys
import argparse
from tokenizer import tokenize

def my_best_segmenter(token_list): 
    all_sentences = []
    this_sentence = []
    for index in range(len(token_list)):
        this_sentence.append(token_list[index])
        if token_list[index] in [ ';', '!', '?']:
            all_sentences.append(this_sentence)
            this_sentence = []

        elif token_list[index] is '.':
            if token_list[index - 1] in [ 'Mr', 'Dr', 'Mrs', 'etc', 'St', 'Gov', "Ave", "Sen"]:
                pass
            elif '.' in token_list[index - 2]:
                pass
            elif len(token_list[index-1]) == 1 and token_list[index - 1].isupper():
                pass
                
            else:
                all_sentences.append(this_sentence)
                this_sentence = []

        

    return all_sentences

def baseline_segmenter(token_list):
    all_sentences = []
    this_sentence = []
    for token in token_list:
        this_sentence.append(token)
        if token in ['.', ':', ';', '!', '?']:
            all_sentences.append(this_sentence)
            this_sentence = []
    return all_sentences

def write_sentence_boundaries(sentence_list, out):
    """ TODO: Write out the token numbers of the sentence boundaries. """
    #start the index counter at -1 because python is 0-indexed allowing to use length
    #for all tokens because it will now be the index of the last token
    indexCounter = -1

    for sentence in sentence_list:
        indexCounter += len(sentence)
        out.write(str(indexCounter) + '\n')
        





def main(args):
    tokens = []
    for line in args.textfile:
        tokens += tokenize(line)
    
    segmentedSentences = my_best_segmenter(tokens)
    sb = write_sentence_boundaries(segmentedSentences, args.hypothesis_file)
    



"""You may have opened a file using something like
        with open(file_name) as file_pointer:
or
        file_pointer = open(file_name, 'w')
where file_pointer would refer to a pointer to some place in memory with a
file.  As you read or write information, the location of that pointer in memory
will change. This is different from how we often think of opening a file, as
the program won't necessarily know the whole contents of the file when you
first open it. In situations where we have huge files with many lines, we can
use this to loop through line by line without having to keep everything in
memory at once.

In order to get lines from the file, we can say
        file_pointer.read_line()
or iterate through lines like
        for line in file_pointer:

Throughout this class, we'll have ArgumentParsers helping to pass in files.  If
you tell an ArgumentParser that a passed-in string is a path to a filename, it
can open the file for you: for instance, --textfile below will allow us to
access the file pointer (not just the filename) as the variable args.textfile.
To do this, you also need to specify the mode to open the file: here, the
hypothesis file is open under write mode 'w', unlike the text file, which is
open in read mode 'r'. If you use this functionality, you shouldn't ever call
the open command yourself: you should tread the passed object the same way as
you treated file_pointer in the snippets above.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentence Segmenter for NLP Lab")
    parser.add_argument('--textfile', "-t", metavar="FILE", type=argparse.FileType('r'),
                        required=True, help="Unlabeled text is in FILE.")
    parser.add_argument("--hypothesis_file", "-y", metavar="FILE", type=argparse.FileType('w'),
                        required=False, default=sys.stdout,
                        help="Write hypothesized boundaries to FILE")

    args = parser.parse_args()
    main(args)
