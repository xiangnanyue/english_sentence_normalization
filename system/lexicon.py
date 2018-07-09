# py2.7
# this script generates a lexicon for the texts to be normalized
import os
import pickle

def import_dictionary(txt_path, pkl_path='words.pkl'):
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            dic = pickle.load(f)
    else:
        dic = dict()
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                word = line.strip().lower()
                dic[word] = 0
        with open(pkl_path, 'wb') as hdl:
            pickle.dump(dic, hdl, protocol=pickle.HIGHEST_PROTOCOL)
    return dic


def freq_dic(train_dir, maxline=10000):
    # return the frequency dictionary for the training txt.
    # the dictionary will be used as a prior knowledge 
    pass