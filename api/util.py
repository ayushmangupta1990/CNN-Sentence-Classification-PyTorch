import logging
import re
import string

def clean_str(string):
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
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().split()

def load_txt_and_tokenize(corpus_path):
    """
        load corpus and remove stopwords and return tokenized corpus
        
        Args:
            corpus_path(list) : the list of path of corpus
            
        Return:
            tokenized corpus with list type
    """
    tokenized_corpus = list()
    for path in corpus_path:
        with open(path) as f:
            for line in f:
                line = clean_str(line.lower().strip())
                for word in line:
                    tokenized_corpus.append(word)
    f.close()                    
    return tokenized_corpus

class LoggerClass():
    def __init__(self, logfilepath):
        super(LoggerClass, self).__init__()
        self.logger = logging.getLogger('mylogger')
        fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
        fileHandler = logging.FileHandler(logfilepath)
        streamHandler = logging.StreamHandler()
        fileHandler.setFormatter(fomatter)
        streamHandler.setFormatter(fomatter)
        self.logger.addHandler(fileHandler)
        self.logger.addHandler(streamHandler)
        self.logger.setLevel(logging.DEBUG)