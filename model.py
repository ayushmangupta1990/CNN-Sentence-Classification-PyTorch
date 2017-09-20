class SentenceClassifier(nn.Module):
    
    def __init__(self, args):
        super(SentenceClassifier, self).__init__()
        self.args = args # parameters
        V = args.UNIQUE_WORD_SIZE # unique word size
        D = args.EMBED_SIZE # word embedding dimension size
        C = args.CNN_CLASS_NUM # classification class type num
        Ci = 1 # the number of channels in the input
        Co = args.CNN_OUTPUT_CHANNEL_NUM # Number of channels produced by the convolution
        Ns = args.CNN_N_GRAM_LIST # list of the number of words (N-gram) EX. [2 3 5]
        self.embed = nn.Embedding(V, D)
        self.embed.weight.data.copy_(torch.from_numpy(args.EMBED_WEIGHT))
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (N, D)) for N in Ns])
        self.dropout = nn.Dropout(args.CNN_DROPOUT_RATE) # the probability for dropout
        self.fcs = nn.Linear(len(Ns)*Co, C)

    def forward(self, x):
        x = self.embed(x) # (N,W,D)
        if self.args.CNN_EMBED_STATIC:
            x = Variable(x)
        x = x.unsqueeze(1) # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x) # (N,len(Ks)*Co)
        return self.fcs(x) # (N,C), logit