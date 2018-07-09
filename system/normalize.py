# python2.7

###### define the parametres #######
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--test', default="True",
        help='test with --head lines in the corpus')
parser.add_argument('--head', default=10)
parser.add_argument('--output_dir', default='./output.txt',
        help='output the normalized text to output_dir')
parser.add_argument('--model_dir', 
        default='./context2vec.ukwac.model.package/context2vec.ukwac.model.params')
parser.add_argument('--train_dir', default='CorpusBataclan_en.1M.raw.txt')
parser.add_argument('--dictionary_dir', type=str, default='./words.txt')

args = parser.parse_args()

import unicodedata


def strip_accents_unicode(s):
    try:
        s = unicode(s, 'utf-8')
    except NameError:  # unicode is a default on python 3
        pass
    s = unicodedata.normalize('NFD', s)
    s = s.encode('ascii', 'ignore')
    s = s.decode("utf-8")
    return str(s)

###### import the nltk words dictionary ########
import nltk
nltk.download('words')
from nltk.corpus import words
from lexicon import import_dictionary

# create a dictionary from the nltk words.words()
# which should contain most of the english words
if args.dictionary_dir is None:
    dictionary = dict.fromkeys(words.words(), None)
else:
    dictionary = import_dictionary(args.dictionary_dir)

# a funciton to search for the word
def is_word(word):
    try:
        x = dictionary[word]
        return True
    except KeyError:
        return False

####### define the context information model ######
import numpy as np
import six
import sys
import traceback
import re

from chainer import cuda
from context2vec.common.context_models import Toks
from context2vec.common.model_reader import ModelReader
from Levenshtein import Levenshtein_Distcance

http_exp = re.compile('https:\/\/[^\s]+')
num_exp = re.compile('[-+]?\d*\.\d+|\d+')

# clean text 
import string 
import nltk
from nltk.stem import SnowballStemmer
ps = SnowballStemmer('english')

punctuation = set(string.punctuation)
punctuation.update(["``", "`", "..."])
punctuation.remove("'")
punctuation.remove("-")
punctuation.remove("@")
punctuation.remove(":")
punctuation.remove(".")

def clean_text(line):
    line = strip_accents_unicode(line)
    # find and remove the https://...
    m = re.search(http_exp, line)
    if m is not None:
        line = line[:m.start()] + line[m.end():]
    line = ''.join(ch for ch in line if ch not in punctuation)
    words = [w.lower() for w in nltk.word_tokenize(line)]
    #stemmed_words = [ps.stem(word) for word in words]
    return words
    
# we enter the words - already stripped ones, 
# and is_words - a list of booleans
def parse_input(words):
    is_words = list(map(lambda x: is_word(x), words))

    target_pos = []
    for i, word in enumerate(words):
        if is_words[i] == False \
            and word not in string.punctuation \
            and re.match(num_exp, word) is None:
            target_pos.append(i)
            
    return words, target_pos
    

def mult_sim(w, target_v, context_v):
    target_similarity = w.dot(target_v)
    target_similarity[target_similarity<0] = 0.0
    context_similarity = w.dot(context_v)
    context_similarity[context_similarity<0] = 0.0
    return (target_similarity * context_similarity)
 

model_param_file = args.model_dir
n_result = 10  # number of search result to show

model_reader = ModelReader(model_param_file)
w = model_reader.w
word2index = model_reader.word2index
index2word = model_reader.index2word
model = model_reader.model

def similarity_matrix(sent, targets):
    
    if len(targets) == 0:
        print "All words true, Can't find the target position"
        return {}

    # the table to return 
    similarity_table = dict()

    for target_pos in targets:
        # in some cases, the word cannot be found in the vocab
        if sent[target_pos] == None:
            target_v = None
        elif sent[target_pos] not in word2index:
            print "Target word", sent[target_pos], " is out of vocabulary."
            target_v = None
        else:
            target_v = w[word2index[sent[target_pos]]]
        
        # calculate the context vector in case ou
        if len(sent) > 1:
            context_v = model.context2vec(sent, target_pos) 
            context_v = context_v / np.sqrt((context_v * context_v).sum())
        else:
            context_v = None
        
        if target_v is None and context_v is None:
            print "Can't find a target nor context."
            similarity_table[target_pos] = {}
            continue

        if target_v is not None and context_v is not None:
            similarity = mult_sim(w, target_v, context_v)
        else:
            if target_v is not None:
                v = target_v
            elif context_v is not None:
                v = context_v                
            similarity = (w.dot(v)+1.0)/2 # Cosine similarity can be negative, mapping similarity to [0,1]

        count = 0
        dic = dict()
        for i in (-similarity).argsort():
            if np.isnan(similarity[i]):
                continue
            #print '{0}: {1}'.format(index2word[i], similarity[i])
            # check the levenshtein distance : cannot be too far away
            # the base sometimes misses the correct word: try better base
            if Levenshtein_Distcance(sent[target_pos], index2word[i]) <= 2:
                dic[index2word[i]] = similarity[i]
            count += 1
            if count == n_result:
                break
        similarity_table[target_pos] = dic
    return similarity_table

    
####### iterate through the files #######
# create the input file object and the output file object
f = open(args.train_dir, 'r')
f_out = open(args.output_dir, 'w')

# maximum number of lines to read
iter_n = 2e6
if args.test == "True":
    iter_n = int(args.head)

index = 0

while index <= iter_n:
    try :
        line = f.readline()
    except:
        break
    words = clean_text(line)
    sent, targets = parse_input(words)
    print sent
    print targets
    table = similarity_matrix(sent, targets)
    print table
    # update sent 
    for key in table.keys():
        if len(table[key]) > 0:
            sent[key] = table[key].keys()[0]
    newline =  ' '.join(sent)
    print newline
    f_out.write(newline + '\n')

    index += 1

f.close()
f_out.close()