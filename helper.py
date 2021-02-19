import numpy as np 
import re
import itertools
from collections import Counter
from tensorflow.contrib import learn
import gensim

class helper:
    def __init__(self):
        embedding_path = '/tempspace/hyuan/data_text/GoogleNews-vectors-negative300.bin'
        self.w = gensim.models.KeyedVectors.load_word2vec_format(embedding_path , binary= True)


    def find_neighbors(self, vec, k): # find the nn based on cos similarity.
        return self.w.similar_by_vector(vec, k)
    
    def get_vector(self, word):
        return self.w[word]



    