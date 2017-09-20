import os
import tarfile
import zipfile
from six.moves.urllib import request
import logging
import re
import string
import numpy as np

class MovieReviewDataset():
    """
        Handling movie review dataset
        
        Dataset
            Movie Review Data
            Available at : http://www.cs.cornell.edu/people/pabo/movie-review-data/
        
        Example
            >> MR = MovieReview()
            >> MR.raw_data["pos"][:10] # get 10 examples
    """
    def __init__(self, encoding):
        """
            Download dataset and parsing it
        """
        if not os.path.exists("./data/"):
            os.makedirs("./data/")
        self.download_file("http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz","./data/rt-polaritydata.tar.gz")
        self.extract_file("./data/rt-polaritydata.tar.gz")
        self.CORPUS_PATH = ["./data/rt-polaritydata/rt-polarity.pos", "./data/rt-polaritydata/rt-polarity.neg"]
        self.processed_corpus_path = self.preprocessing(CORPUS_PATH, encoding)
        self.tokenized_corpus = self.tokenize_all(processed_corpus_path)
        self.unique_word_list = np.unique(self.tokenized_corpus) 
        self.unique_word_list_size = self.unique_word_list.size   
        self.word_to_index = {word: index for index, word in enumerate(self.unique_word_list)}

    def download_file(self, url, path):
        """
            Downlaod file at path
        """
        if not os.path.exists(path):
            request.urlretrieve(url, path)     

    def extract_file(self, path, to_directory="./data/"):
        """
            Extract file
        """
        if path.endswith('.zip'):
            opener, mode = zipfile.ZipFile, 'r'
        elif path.endswith('.tar.gz') or path.endswith('.tgz'):
            opener, mode = tarfile.open, 'r:gz'
        elif path.endswith('.tar.bz2') or path.endswith('.tbz'):
            opener, mode = tarfile.open, 'r:bz2'
        else: 
            raise (ValueError, "Could not extract `%s` as no appropriate extractor is found" % path)

        file = opener(path, mode)
        try: 
            file.extractall(to_directory)
        finally: 
            file.close()

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets.
        """
        # Tips for handling string in python : http://agiantmind.tistory.com/31   
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
        string = re.sub(r"\'s", " \'s", string) 
        string = re.sub(r"\'ve", " \'ve", string) 
        string = re.sub(r"n\'t", " n\'t", string) 
        string = re.sub(r"\'re", " \'re", string) 
        string = re.sub(r"\'d", " \'d", string) 
        string = re.sub(r"\'ll", " \'ll", string) 
        string = re.sub(r",", " , ", string) 
        string = re.sub(r"!", " ! ", string) 
        string = re.sub(r"\(", " \( ", string) 
        string = re.sub(r"\)", " \) ", string) 
        string = re.sub(r"\?", " \? ", string) 
        string = re.sub(r"\s{2,}", " ", string) # replace more than 2 whitespace with 1 whitespace   
        return string.strip().split()

    def proprocessing(self, corpus_path, encoding):
        # UnicodeDecodeError: 'utf-8' codec can't decode byte
        # https://stackoverflow.com/questions/19699367/unicodedecodeerror-utf-8-codec-cant-decode-byte
        # get max line length
        mx = 0
        for path in corpus_path:
            with open(path, encoding=encoding) as f:
                for line in f:
                    line = clean_str(line.lower().strip())
                    if len(line) > mx: 
                        mx = len(line)
            f.close()   
        processed_corpus_path = list()  
        for path in corpus_path:
            new_path = path + ".processed"
            processed_corpus_path.append(new_path)
            with open(path, encoding=encoding) as f:
                with open(new_path, 'w') as f_new:
                    for line in f:
                        new_line = clean_str(line.lower().strip())
                        new_line += " <PAD>"*(mx-len(new_line))
                        f_new.write(new_line+'\n')
                        break # for dubug
                f_new.close() 
            f.close()               
        return processed_corpus_path

    def tokenize_all(self, processed_corpus_path):
        """
            return tokenized corpus
        """
        tokenized_corpus = list()
        for path in processed_corpus_path:
            with open(path) as f:
                for line in f:
                    for word in line.strip().split():
                        self.tokenized_corpus.append(word)
            f.close()
        return tokenized_corpus

    def build_train_data(self):
        train_data = list()
        with open("./data/rt-polaritydata/rt-polarity.pos.processed") as f:
            train_pos_data = np.array([[[word_to_index[word]] for word in line.strip().split()] for line in f])
            train_pos_label = np.ones(train_pos_data.size)
        f.close()
        with open("./data/rt-polaritydata/rt-polarity.neg.processed") as f:
            train_neg_data = np.array([[[word_to_index[word]] for word in line.strip().split()] for line in f])
            train_neg_label = np.zeros(train_neg_data.size)
        f.close()
        datasize = train_pos_data.size + train_neg_data.size
        return train_data, train_label, datasize