import numpy as np
import re
import itertools
from collections import Counter
from tensorflow.contrib import learn
import csv
from html.parser import HTMLParser
import scipy
import time
import operator

class data_reader:
    def __init__(self):
        print("Now loading data ==================")
        self.path_train= '/tempspace/hyuan/data_text/ag_news_csv/train.csv'
        self.path_test= '/tempspace/hyuan/data_text/ag_news_csv/test.csv'
        self.data_path = '/tempspace/hyuan/text_interpretation_test/new_representation.txt'
        self.x_train, self.y_train = self.get_datasets(self.path_train)
        self.x_dev, self.y_dev = self.get_datasets(self.path_test)
        self.x_test_visual = self.read_data_visal(self.data_path )


        self.x_train, self.y_train = self.load_data_labels(self.x_train, self.y_train)
        self.x_dev, self.y_dev = self.load_data_labels(self.x_dev, self.y_dev)
        self.x_dev_text= self.x_dev
        # for i in range(186,190):
        #     print(self.x_train[i])
        self.length = [len(x.split(" ")) for x in self.x_train]
        self.length_dev = [len(x.split(" ")) for x in self.x_dev]
        self.length_visual = [len(x.split(" ")) for x in self.x_test_visual]

        self.max_document_length = max(self.length)
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(self.max_document_length, min_frequency=0)

        self.mask_train = np.array(self.get_mask(self.length))  
        self.mask_dev  = np.array(self.get_mask(self.length_dev)) 
        self.mask_visual = np.array(self.get_mask(self.length_visual))

        self.vocab = self.vocab_processor.fit((self.x_train))

        self.x_train = np.array(list(self.vocab.transform(self.x_train)))    

        self.x_dev = np.array(list(self.vocab.transform(self.x_dev)))
        self.x_test_visual = np.array(list(self.vocab.transform(self.x_test_visual)))

        print("Vocabulary Size: {:d}".format(len(self.vocab_processor.vocabulary_)))
        print("Train/Dev split: {:d}/{:d}".format(len(self.y_train), len(self.y_dev)))    
        print("the max document length is: ", self.max_document_length)
        print("Finished loading data ==================")
        self.test_size =  len(self.y_dev)
        self.gen_index()
        self.train_idx = 0 
        self.test_idx = 0

    def read_data_visal(self, data_path ):
        res = []
        with open(data_path, "r") as f:
            lines = f.read().splitlines() 
            for line in lines:
                text = line.split(',')[-1]
                res.append(text)
        return res

    def read_visual_data(self):
        return self.x_test_visual, self.y_dev[0:1000], self.mask_visual
    


    def get_mask(self, l):
        res = []
        for item in l:
            msk = [1]*item + [0]*(self.max_document_length -item)
            res.append(np.array(msk))
        return res


    def convert_to_text(self, x):
        return np.array(list(self.vocab_processor.reverse(x)))

    def get_word_dict(self):
        w_dict = {v:k for k,v in self.vocab_processor.vocabulary_._mapping.items()}
        return w_dict

    def get_datasets_20newsgroup(self, shuffle=True, random_state=42):

        datasets = fetch_20newsgroups(shuffle=shuffle, random_state=random_state)
        return datasets

    def gen_index(self):
        self.indexes = np.random.permutation(range(len(self.y_train)))
        self.train_idx = 0

    def get_test_x(self):
     #   k = np.random.randint(self.test_size)
        k = 16
        return self.x_dev[k:k+1], self.x_dev_text[k:k+1], self.mask_dev[k:k+1], self.y_dev[k:k+1]

    def get_test_x2(self):
        return self.x_dev[1:2]

    def next_batch(self, batch_size):
        next_index = self.train_idx + batch_size
        cur_indexes = list(self.indexes[self.train_idx:next_index])
        self.train_idx = next_index
        if len(cur_indexes) < batch_size:
            self.gen_index()
            return self.next_batch(batch_size)
        cur_indexes.sort()
        return self.x_train[cur_indexes], self.y_train[cur_indexes], self.mask_train[cur_indexes]   

    def get_random_test(self):
        k = np.random.randint(self.test_size)
        return self.x_dev[k:k+1], self.y_dev[k:k+1], self.x_dev_text[k:k+1]
    
    def next_test_batch(self,batch_size):
        prev_idx = self.test_idx
        self.test_idx += batch_size
        if self.test_idx > self.test_size:
            self.test_idx = self.test_size
            prev_idx = self.test_idx - batch_size
            self.reset()
      #  print(prev_idx, "to ===========", self.test_idx)
        return self.x_dev[prev_idx:self.test_idx], self.y_dev[prev_idx:self.test_idx], self.x_dev_text[prev_idx:self.test_idx], self.mask_dev[prev_idx:self.test_idx]

    def all_test_batch(self):
        return self.x_dev[0:1000], self.y_dev[0:1000], self.mask_dev[0:1000]

    
    def reset(self):
        self.test_idx = 0

    # def clean_str(self,string):
    #     """
    #     Tokenization/string cleaning for all datasets except for SST.
    #     Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    #     """
    #     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    #     string = re.sub(r"\'s", " \'s", string)
    #     string = re.sub(r"\'ve", " \'ve", string)
    #     string = re.sub(r"n\'t", "n\'t", string)
    #     string = re.sub(r"\'re", " \'re", string)
    #     string = re.sub(r"\'d", " \'d", string)
    #     string = re.sub(r"\'ll", " \'ll", string)
    #     string = re.sub(r",", " , ", string)
    #     string = re.sub(r"!", " ! ", string)
    #     string = re.sub(r"\(", " \( ", string)
    #     string = re.sub(r"\)", " \) ", string)
    #     string = re.sub(r"\?", " \? ", string)
    #     string = re.sub(r"\s{2,}", " ", string)
    #     return string.strip().lower()

    def clean_str(self, string):
        '''Tokenization/string cleaning.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        '''

        # replace all kinds of arabic numbers
        string = re.sub(r"#[0-9]*;", " ", string)
        string = re.sub(r"#", " ", string)
        string = re.sub(r"[0-9]+\.[0-9]+", "#", string)
        string = re.sub(r"[0-9]+,[0-9]+", "#", string)
        string = re.sub(r"[0-9]+", "#", string)

        # handle abbreviation like U.S. / U.S.A
        string = re.sub(r"([A-Za-z]+)\.([A-Za-z]+)\.([A-Za-z]+)\.([A-Za-z]+)", r"\1\2\3\4", string)
        string = re.sub(r"([A-Za-z]+)\.([A-Za-z]+)\.([A-Za-z]+)", r"\1\2\3", string)
        string = re.sub(r"([A-Za-z]+)\.([A-Za-z]+)", r"\1\2", string)

        # handle 'word' 'word word'
        string = re.sub(r"([^A-Za-z])\'+([A-Za-z])", r"\1\2", string)
        string = re.sub(r"([A-Za-z])\'+([^A-Za-z])", r"\1\2", string)
        string = re.sub(r"^\'+", r"", string)
        string = re.sub(r"\'+$", r"", string)

        string = re.sub(r"[^A-Za-z0-9(),!?\'\`#]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"#", " # ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip()#.lower()



    def get_datasets(self, path_data):
        '''
        load data and label from two files.
        '''
        x_text= []
        y= []
        html_parser = HTMLParser()
        with open(path_data, "r", encoding='windows-1252') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            for row in rdr:
            #    text= row[1]+' '+row[2]
                text = html_parser.unescape(row[1])+' '+ html_parser.unescape(row[2])
                x_text.append(text)
                y.append(int(row[0]))
        y = np.array(y)
        return[x_text, y]

    def load_data_labels_news(self, datasets):
        x_text = datasets.data
        x_text = [self.clean_str(sent) for sent in x_text]
        # labels = []+-- 
        # for i in range(len(x_text)):
        #     label = [0 for j in datasets['target_names']]
        #     label[datasets['target'][i]] = 1
        #     labels.append(label)
        # y = np.array(labels)
        return [x_text, datasets.target]   

    def load_data_labels(self, x, y):
        x_text = [self.clean_str(sent) for sent in x]
        labels = []
        n_values = np.max(y)
        y = np.eye(n_values)[y-1]
        return [x_text, y]    

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]


    def load_word2vec_embedding(self, vocabulary, filename, binary):
        '''
        Load the word2vec embedding. 
        /tempspace/hyuan/data_text/GoogleNews-vectors-negative300.bin', "the path for pre-trained embedding"
        '''
        encoding = 'utf-8'
        with open(filename, "rb") as f:
            header = f.readline()
            vocab_size, vector_size = map(int, header.split())
            # initial matrix with random uniform
            embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
            if binary:
                binary_len = np.dtype('float32').itemsize * vector_size
                for line_no in range(vocab_size):
                    word = []
                    while True:
                        ch = f.read(1)
                        if ch == b' ':
                            break
                        if ch == b'':
                            raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                        if ch != b'\n':
                            word.append(ch)
                    word = str(b''.join(word), encoding=encoding, errors='strict')
                    idx = vocabulary.get(word)
                    if idx != 0:
                        embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                    else:
                        f.seek(binary_len, 1)
            else:
                for line_no in range(vocab_size):
                    line = f.readline()
                    if line == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                    if len(parts) != vector_size + 1:
                        raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                    word, vector = parts[0], list(map('float32', parts[1:]))
                    idx = vocabulary.get(word)
                    if idx != 0:
                        embedding_vectors[idx] = vector
            f.close()
            return embedding_vectors

    def find_neighbors(self, embedding, vec, k):
        dict_dis={}
        for i in range(len(self.vocab_processor.vocabulary_)):
            cos_dis = scipy.spatial.distance.cosine(vec, embedding[i])
            dict_dis[i]= cos_dis
        
        final_sorted = sorted(dict_dis.items(), key=operator.itemgetter(1), reverse=False)
        res = []
        for i in range(k):
            item = final_sorted[i]
           # print(item[0])
            lst = []
            lst.append(item[0])
         #   print(lst)
            lst2= []
            lst2.append(lst)

         #   print(lst2)
            word = self.convert_to_text(lst2)
         #   print(word)
            res.append((word[0], item[1], embedding[item[0]]))
   #         print(word[0], item[1])
        
      #  print(res[0])
     #   time.sleep(20)
        return res