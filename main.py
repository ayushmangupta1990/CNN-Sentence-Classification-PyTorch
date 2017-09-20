import os
import datetime
import numpy as np
import torch
from torch.autograd import Variable
from progressbar import ProgressBar
from api.util import LoggerClass
from api.model import GloVeClass
from api.util import MovieReviewDataset
from api.util import load_txt_and_tokenize

print = LoggerClass("./20170831.log").logger.info

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/local/lib:/usr/lib64"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
CUDA_AVAILABLE = torch.cuda.is_available()
if not CUDA_AVAILABLE:
	raise Exception("CUDA is not available")
torch.manual_seed(1)
torch.cuda.manual_seed(1)

PARAMS = {    
	"EMBED_SIZE" : 300,
    "Skip_GloVe" : True,

	"GLOVE_CONTEXT_SIZE" : 5,
	"GLOVE_X_MAX" : 100,
	"GLOVE_ALPHA" : 0.75,
	"GLOVE_L_RATE" : 0.05,
	"GLOVE_PROCESS_NUM" : 4,
	"GLOVE_BATCH_SIZE" : 1024,
	"GLOVE_NUM_EPOCHS" : 50,

    "CNN_CLASS_NUM" : 2,
    "CNN_OUTPUT_CHANNEL_NUM" : 5,
    "CNN_N_GRAM_LIST" : [2,3,4,5],
    "CNN_DROPOUT_RATE" : 0.5,
    "CNN_EMBED_STATIC" : False,
    "CNN_L_RATE" : 0.01,
    "CNN_NUM_EPOCHS" : 50,
    "CNN_BATCH_SIZE" : 64
}
print("Parameters : {}".format(PARAMS))

print("Word embedding(GloVe) start")
MR = MovieReviewDataset(encoding = "ISO-8859-1")
PARAM["UNIQUE_WORD_SIZE"] = MR.unique_word_list_size
print("TOKENIZED_CORPUS_SIZE : {}".format(len(MR.tokenized_corpus)))
print("UNIQUE_WORD_SIZE : {}".format(MR.unique_word_list_size))

if not PARAM["Skip_GloVe"]:
    GloVe = GloVeClass(
    	MR.tokenized_corpus, 
    	MR.unique_word_list, 
    	PARAMS["EMBED_SIZE"],  
    	PARAMS["GLOVE_CONTEXT_SIZE"], 
    	PARAMS["GLOVE_X_MAX"], 
    	PARAMS["GLOVE_ALPHA"], 
    	PARAMS["GLOVE_PROCESS_NUM"]
    	)
    GloVe = torch.nn.DataParallel(GloVe, device_ids=[0, 1, 2, 3]).cuda()
    for p in GloVe.parameters():
        print(p.size())
    print("GloVe Training Start")
    optimizer = torch.optim.Adagrad(GloVe.parameters(), PARAMS["GLOVE_L_RATE"])
    for epoch in range(PARAMS["GLOVE_NUM_EPOCHS"]):
        print("Epoch {} start".format(epoch + 1))
        losses = []
        update_time = ((MR.unique_word_list_size * MR.unique_word_list_size) // PARAMS["GLOVE_BATCH_SIZE"])
        p = ProgressBar(maxval = update_time).start()  
        for i in range(1, update_time + 1):
            word_u_variable, word_v_variable, words_co_occurences, words_weights = GloVe.module.next_batch(PARAMS["GLOVE_BATCH_SIZE"])
            optimizer.zero_grad()
            forward_output = GloVe(word_u_variable, word_v_variable)
            loss = (torch.pow((forward_output - torch.log(words_co_occurences)), 2) * words_weights).sum()
            losses.append(loss.data[0])
            loss.backward()
            optimizer.step()
            p.update(i)
        p.finish()
        print("Train Epoch: {} \t Loss: {:.6f}".format(epoch + 1, np.mean(losses)))
        np.savez(
            'model/glove.npz', 
            word_embeddings_array=GloVe.module.embedding(), 
            word_to_index=GloVe.module.word_to_index,
            index_to_word=GloVe.module.index_to_word
            )
    del GloVe
    print("Done")

print("Movie Review Sentence Classification start")
data = np.load('model/glove.npz')
word_embeddings_array = data['word_embeddings_array']
PARAMS["EMBED_WEIGHT"] = word_embeddings_array
CNN = SentenceClassifier(PARAMS)
CNN = torch.nn.DataParallel(CNN, device_ids=[0, 1, 2, 3]).cuda()
for p in CNN.parameters():
    print(p.size())
print("CNN Training Start")
optimizer = torch.optim.Adam(CNN.parameters(), lr=PARAMS["CNN_L_RATE"])
train_data, train_label, datasize = MR.build_train_data()
CNN.train()
for epoch in range(PARAMS["CNN_NUM_EPOCHS"]):
    indexes = np.random.permutation(datasize)
    updatetime = int(datasize/PARAMS["CNN_BATCH_SIZE"])+1
    p = ProgressBar(maxval = updatetime).start()
    for i in range(updatetime):
        pos = i*PARAMS["CNN_BATCH_SIZE"]
        ids = indexes[pos:(pos+PARAMS["CNN_BATCH_SIZE"]) if (pos+PARAMS["CNN_BATCH_SIZE"]) < datasize else datasize]
        batch_train_data = Variable(torch.from_numpy(train_data[ids]).cuda())
        batch_train_label = Variable(torch.from_numpy(train_label[ids]).cuda())
        optimizer.zero_grad()
        logit = model(batch_train_data)
        #print('logit vector', logit.size())
        #print('target vector', target.size())
        loss = F.cross_entropy(logit, batch_train_label)
        loss.backward()
        optimizer.step()