##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##

from abc import ABC, abstractmethod
from itertools import islice
from html import unescape
from scipy import sparse 
from lxml import etree
import sys


#####################################################################
# HELPER FUNCTIONS
#####################################################################

def do_xml_parse(fp, tag, max_elements=None, progress_message=None):
    """ 
    Parses cleaned up spacy-processed XML files
    """
    fp.seek(0)

    elements = enumerate(islice(etree.iterparse(fp, tag=tag), max_elements))
    for i, (event, elem) in elements:
        yield elem
        elem.clear()
        if progress_message and (i % 1000 == 0): 
            print(progress_message.format(i), file=sys.stderr, end='\r')
    if progress_message: print(file=sys.stderr)

def short_xml_parse(fp, tag, max_elements=None): 
    """ 
    Parses cleaned up spacy-processed XML files (but not very well)
    """
    elements = etree.parse(fp).findall(tag)
    N = max_elements if max_elements is not None else len(elements)
    return elements[:N]

#####################################################################
# PCLVocab
#####################################################################

class PCLVocab():
    '''creates a vocab using the vocab file, and determines size with vocab size.'''
    def __init__(self, vocab_file, vocab_size, num_stop_words): 
        start_index = 0 if num_stop_words is None else num_stop_words
        end_index = start_index + vocab_size if vocab_size is not None else None

        self._words = [w.strip() for w in islice(vocab_file, start_index, end_index)]

        #maps the word as a key and the index as the value
        self._dict = dict([(w, i) for (i, w) in enumerate(self._words)])

    def __len__(self):
        '''returns the length of the vocab'''
        return len(self._dict)

    def index_to_label(self, i): 
        return self._words[i]

    def __getitem__(self, key):
        if key in self._dict: return self._dict[key]
        else: return None

#####################################################################
# PCLLabels
#####################################################################

class PCLLabels(ABC):
    '''creates a list of labels and a dict mappings for the labels using process'''
    def __init__(self): 
        self.labels = None
        self._label_list = None

    def __getitem__(self, index):
        """ return the label at this index """
        return self._label_list[index]

    def process(self, label_file, max_instances=None):
        '''y_labeled is a list of all of the labels in the xml label_file extracted using ._extract_label '''
        y_labeled = list(map(self._extract_label, do_xml_parse(label_file, 'example', max_elements=max_instances)))
        if self.labels is None:
            self._label_list = sorted(set(y_labeled))
            self.labels = dict([(x,i) for (i,x) in enumerate(self._label_list)])
            
        y = [self.labels[x] for x in y_labeled]
        return y

    @abstractmethod        
    def _extract_label(self, example):
        """ Return the label for this instance """
        return "Unknown"

#####################################################################
# PCLFeatures
#####################################################################

class PCLFeatures(ABC):
    """Lots of helper methods for the features a.k.a the vocab"""
    def __init__(self, vocab):
        self.vocab = vocab

    def extract_text(self, example):
        '''returns a list of the tokenized text'''
        return unescape("".join([x for x in example.itertext()]).lower()).split()

    def process(self, data_file, max_instances=None):
        '''processes the data file returning a sparse lil matrix for the features in the data file and the ids of the example'''
        if max_instances == None:
            N = len([1 for example in do_xml_parse(data_file, 'example')])
        else:
            N = max_instances

        X = sparse.lil_matrix((N, self._get_num_features()), dtype='uint8')
        
        ids = []
        example_generator = enumerate(do_xml_parse(data_file, 'example', max_elements=N, progress_message="Example {}"))
        for i, example in example_generator:
            ids.append(example.get("id"))
            for j, value in self._extract_features(example):
                X[i,j] = value
        return X, ids

    @abstractmethod
    def __getitem__(self, i):
        """ Returns an interpretable label for the feature at index i """
        return "Unknown"

    @abstractmethod            
    def _extract_features(self, example):
        """ Returns a list of the features in the example """
        return []

    @abstractmethod        
    def _get_num_features(self):
        """ Return the total number of features """
        return -1

#####################################################################