import os
import datetime
import numpy as np
import torch
from progressbar import ProgressBar
from api.util import LoggerClass
from api.model import GloVeClass
from api.util import load_txt_and_tokenize

print = LoggerClass("./20170830.log").logger.info
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/local/lib:/usr/lib64"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
CUDA_AVAILABLE = torch.cuda.is_available()
if not CUDA_AVAILABLE:
	raise Exception("CUDA is not available")
torch.manual_seed(1)
torch.cuda.manual_seed(1)
PARAMS = {    
	"WORD_EMBED_SIZE" : 300,

	"GLOVE_CONTEXT_SIZE" : 5,
	"GLOVE_X_MAX" : 100,
	"GLOVE_ALPHA" : 0.75,
	"GLOVE_L_RATE" : 0.05,
	"GLOVE_PROCESS_NUM" : 4,
	"GLOVE_BATCH_SIZE" : 1024,
	"GLOVE_NUM_EPOCHS" : 100

}
print("Parameters : {}".format(PARAMS))

print("Word embedding(GloVe) start")
CORPUS_PATH = ["./data/rt-polaritydata/rt-polarity.pos", "./data/rt-polaritydata/rt-polarity.neg"]
tokenized_corpus = load_txt_and_tokenize(CORPUS_PATH, "ISO-8859-1")
unique_word_list = np.unique(tokenized_corpus)
unique_word_list_size = unique_word_list.size
GloVe = GloVeClass(
	tokenized_corpus, 
	unique_word_list, 
	PARAMS["WORD_EMBED_SIZE"], 
	PARAMS["GLOVE_CONTEXT_SIZE"], 
	PARAMS["GLOVE_X_MAX"], 
	PARAMS["GLOVE_ALPHA"], 
	PARAMS["GLOVE_PROCESS_NUM"]
	)
GloVe = torch.nn.DataParallel(GloVe, device_ids=[0, 1, 2, 3]).cuda()
for p in GloVe.parameters():
    print(p.size())
optimizer = torch.optim.Adagrad(GloVe.parameters(), PARAMS["GLOVE_L_RATE"])
for epoch in range(PARAMS["GLOVE_NUM_EPOCHS"]):
    print("Epoch {} start".format(epoch + 1))
    losses = []
    update_time = ((unique_word_list_size * unique_word_list_size) // PARAMS["GLOVE_BATCH_SIZE"])
    p = ProgressBar(maxval = update_time).start()  
    for i in range(1, update_time + 1):
        word_u_variable, word_v_variable, words_co_occurences, words_weights = GloVe.module.next_batch(PARAMS["GLOVE_BATCH_SIZE"])
        forward_output = GloVe(word_u_variable, word_v_variable)
        loss = (torch.pow((forward_output - torch.log(words_co_occurences)), 2) * words_weights).sum()
        losses.append(loss.data[0])
        optimizer.zero_grad()
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
print("Done")

