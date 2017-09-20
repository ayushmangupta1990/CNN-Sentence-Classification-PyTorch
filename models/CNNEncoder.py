import torch
import torch.nn as nn
from torch.autograd import Variable

class CNNEncoder(nn.Module):
    
    def __init__(self, config, word_embedding_array):
        super(CNNEncoder, self).__init__()
        self.config = config 
        self.embed = nn.Embedding(self.config.unique_word_size, self.config.word_edim)
        self.embed.weight.data.copy_(torch.from_numpy(word_embedding_array))
        self.convs = nn.ModuleList([nn.Conv2d(1, self.config.cnn_output_channel, (n, self.config.word_edim)) for n in self.config.cnn_n_gram_list])

    def forward(self, x):
        x = self.embed(x) # (batch_size,seq_len,word_edim)
        if self.config.embed_static:
            x = Variable(x)
        x = x.unsqueeze(1) # (batch_size,1,seq_len,word_edim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] #[(batch_size,cnn_output_channel,seq_len), ...]*len(cnn_n_gram_list)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(batch_size,cnn_output_channel), ...]*len(cnn_n_gram_list)
        x = torch.cat(x, 1)
        return x