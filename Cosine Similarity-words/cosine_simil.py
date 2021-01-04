from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

import urllib.request
import collections
import os
import zipfile

import numpy as np
import tensorflow as tf

def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similarity between u and v
        
    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """
    
    distance = 0.0
    
    dot = np.dot(u,v)

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    cosine_similarity = dot/(norm_u*norm_v)
    
    return cosine_similarity

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task: a is to b as c is to ____. 
    
    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors. 
    
    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """
    
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    
    e_a, e_b, e_c = word_to_vec_map.get(word_a), word_to_vec_map.get(word_b), word_to_vec_map.get(word_c)
    
    words = word_to_vec_map.keys()
    max_cosine_sim = -100              
    best_word = None                  


    input_words_set = set([word_a, word_b, word_c])
    
    for w in words:        
        
        if w in input_words_set:
            continue
        
        cosine_sim = cosine_similarity(np.subtract(e_b,e_a), np.subtract(word_to_vec_map.get(w),e_c))
        
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
         
    return best_word

def main():

    root_path = os.path.dirname(os.path.abspath(__file__))
    #full_path = os.path.join(root_path, 'glove.6B.txt')
    #words, word_to_vec_map = read_glove_vecs(full_path)


window_size = 3
vector_dim = 300
epochs = 1000

valid_size = 16     
valid_window = 100  
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

def maybe_download(filename, url, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def collect_data(vocabulary_size=10000):
    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download('text8.zip', url, 31344016)
    vocabulary = read_data(filename)
    print(vocabulary[:7])
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                                vocabulary_size)
    del vocabulary  
    return data, count, dictionary, reverse_dictionary

class SimilarityCallback:
    def run_sim(self):
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        for i in range(vocab_size):
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim
    

def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec_map

def relu(x):

    s = np.maximum(0,x)
    return s


def initialize_parameters(vocab_size, n_h):
    
    parameters = {}

    parameters['W1'] = np.random.randn(n_h, vocab_size) / np.sqrt(vocab_size)
    parameters['b1'] = np.zeros((n_h, 1))
    parameters['W2'] = np.random.randn(vocab_size, n_h) / np.sqrt(n_h)
    parameters['b2'] = np.zeros((vocab_size, 1))

    return parameters

def softmax(x):

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

if __name__ = "__main__":
    main()