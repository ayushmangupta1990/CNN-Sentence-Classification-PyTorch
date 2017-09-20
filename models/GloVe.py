import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import print

import datetime
import numpy as np
from scipy.sparse import coo_matrix
import multiprocessing as mp
import torch
import torch.nn as nn
from torch import optim
from torch.nn.init import xavier_normal
from torch.autograd import Variable
from utils import ProgressBar

class GloVe(nn.Module):
    """
        GloVe

        Parameters
            config(argparse.Namespace): parameters and configs captured by parser
            tokenized_corpus(list): all tokenized corpus
            dictionary(dict): 
                word2idx(dict)
                idx2word(dict)
    """
    def __init__(self, config, tokenized_corpus, dictionary):
        super(GloVe, self).__init__()
        self.config = config
        self.build_model(tokenized_corpus, dictionary)

    def build_model(self, tokenized_corpus, dictionary):
        self.tokenized_corpus = tokenized_corpus
        self.dictionary = dictionary
        self.tokenized_corpus_size = len(self.tokenized_corpus)

        self.in_embed = nn.Embedding(self.config.unique_word_size, self.config.word_edim)
        self.in_embed.weight = xavier_normal(self.in_embed.weight)
        self.in_bias = nn.Embedding(self.config.unique_word_size, 1)
        self.in_bias.weight = xavier_normal(self.in_bias.weight)
        self.out_embed = nn.Embedding(self.config.unique_word_size, self.config.word_edim)
        self.out_embed.weight = xavier_normal(self.out_embed.weight)
        self.out_bias = nn.Embedding(self.config.unique_word_size, 1)
        self.out_bias.weight = xavier_normal(self.out_bias.weight)  

        self.word_embedding_array = None
        self.word_u_candidate = np.arange(self.config.unique_word_size)
        self.word_v_candidate = np.arange(self.config.unique_word_size)  

        # Count co-occurence with multiple process
        queue = mp.Queue()
        ps = list()
        for i in range(self.config.glove_process_num):
            ps.append(mp.Process(target=self.build_sub_co_occurence_matrix, args=(queue, i)))
        for p in ps:
            p.start()
        # キューから結果を回収
        for i in range(self.config.glove_process_num):
            if i:
                col += queue.get()   # キューに値が無い場合は、値が入るまで待機になる
            else:
                col = queue.get()
        for p in ps:
            p.terminate()
        col = np.array(col, dtype = np.int64)
        self.co_occurence_matrix = coo_matrix(
            (np.ones(col.size, dtype = np.int64), (np.zeros(col.size, dtype = np.int64), col)), 
            shape=(1, int((self.config.unique_word_size * (self.config.unique_word_size + 1)) / 2)),
            dtype = np.int64
        )
        self.co_occurence_matrix = self.co_occurence_matrix.todense()

    def build_sub_co_occurence_matrix(self, queue, process_num):
        col = list()
        # iの範囲を設定
        ini = int(self.tokenized_corpus_size * process_num / self.config.glove_process_num)
        fin = int(self.tokenized_corpus_size * (process_num + 1) / self.config.glove_process_num)
        for i in range(ini, fin):
            index = self.dictionary["word2idx"][self.tokenized_corpus[i]]
            for j in range(1, self.config.glove_context_size + 1):
                if i - j > 0:
                    left_index = self.dictionary["word2idx"][self.tokenized_corpus[i - j]]
                    col.append(self.convert_pairs_to_index_include_diag(left_index, index))
                if i + j < self.tokenized_corpus_size:
                    right_index = self.dictionary["word2idx"][self.tokenized_corpus[i + j]]
                    col.append(self.convert_pairs_to_index_include_diag(right_index, index))
        queue.put(col)
    
    def weight_func(self, x):
        return 1 if x > self.config.glove_x_max else (x / self.config.glove_x_max) ** self.config.glove_alpha

    def convert_pairs_to_index_include_diag(self, word_u_index, word_v_index):
        u = min(word_u_index, word_v_index)
        v = max(word_u_index, word_v_index)
        # return int((unique_word_size + (unique_word_size - (u - 1))) * u / 2 + (v - u))
        return int((2 * self.config.unique_word_size - u + 1) * u / 2 + v - u)

    def next_batch(self):
        batch_size = self.config.glove_batch_size
        word_u = np.random.choice(self.word_u_candidate, size=batch_size)
        word_v = np.random.choice(self.word_v_candidate, size=batch_size)
        # + 1 -> to prevent having log(0)
        words_co_occurences = np.array(
            [self.co_occurence_matrix[0, self.convert_pairs_to_index_include_diag(word_u[i], word_v[i])] + 1 for i in range(batch_size)]
        )
        words_weights = np.array([self.weight_func(var) for var in words_co_occurences])
        if self.config.cuda:
            return Variable(torch.from_numpy(word_u).cuda()), Variable(torch.from_numpy(word_v).cuda()), Variable(torch.from_numpy(words_co_occurences).cuda()).float(), Variable(torch.from_numpy(words_weights).cuda()).float()
        else:
            return Variable(torch.from_numpy(word_u)), Variable(torch.from_numpy(word_v)), Variable(torch.from_numpy(words_co_occurences)).float(), Variable(torch.from_numpy(words_weights)).float()

    def forward(self, word_u, word_v):
        word_u_embed = self.in_embed(word_u)
        word_u_bias = self.in_bias(word_u)
        word_v_embed = self.out_embed(word_v)
        word_v_bias = self.out_bias(word_v)
        return ((word_u_embed * word_v_embed).sum(1) + word_u_bias + word_v_bias).squeeze(1)
    
    def embedding(self):
        self.word_embedding_array = self.in_embed.weight.data.cpu().numpy() + self.out_embed.weight.data.cpu().numpy()
        return self.word_embedding_array

def run_GloVe(config, model):
    if config.show_progress:
        print("Glove model parameters")
        for p in model.parameters():
            print(p.size())
    optimizer = optim.Adagrad(model.parameters(), config.glove_lr)
    if config.show_progress:
        print("Train start")
        print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    for epoch in range(config.glove_epochs):
        N = ((config.unique_word_size*config.unique_word_size) // config.glove_batch_size)
        losses = []
        if config.show_progress:
            bar = ProgressBar('Glove Train epoch {} / {}'.format(epoch+1,config.glove_epochs), max=N)
        for i in range(N):
            if config.cuda and config.gpu_num > 1:
                word_u_variable, word_v_variable, words_co_occurences, words_weights = model.module.next_batch()
            else:
                word_u_variable, word_v_variable, words_co_occurences, words_weights = model.next_batch()
            forward_output = model(word_u_variable, word_v_variable)
            loss = (torch.pow((forward_output - torch.log(words_co_occurences)), 2) * words_weights).sum()
            losses.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if config.show_progress:
                bar.next()
        if config.show_progress:
            bar.finish()
            print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch + 1, np.mean(losses)))
        if config.cuda and config.gpu_num > 1:
            np.savez('./checkpoints/word_embedding.npz', 
                word_embedding_array=model.module.embedding(), 
                dictionary=model.module.dictionary
            )
        else:
            np.savez('./checkpoints/word_embedding.npz', 
                word_embedding_array=model.embedding(), 
                dictionary=model.dictionary
            )