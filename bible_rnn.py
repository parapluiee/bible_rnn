import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize
nltk.download('gutenberg')
from nltk.corpus import gutenberg

def preprocess_sents(sents):
    output = list()
    for i in range(len(sents)):
        output.append(list())
        for j in range(len(sents[i])):
            output[-1].append(sents[i][j].lower())
    return output

def word_list(sents):
    words = set()
    for sent in sents:
        for word in sent:
            words.add(word.lower())
    return list(words)


def vectorize_sents(sents, word_to_vect, dict_size, max_len):
    vector_sents = list()
    i = 0
    
    for sent in sents:
        i+=1
        vector_sents.append([[0] * dict_size] * max_len)
        for w in range(max_len):
            if w < len(sent):
                vector_sents[-1][w] =  word_to_vect[sent[w]]
        if i % 100 == 0:
            print(str(i) + "/" + str(len(sents)) + " Sentences Processed")
    return vector_sents

def input_output(sents):
    output_sents = list()
    input_sents = list()
  
    for sent in sents:
        input_sents.append(sent[:-1])
        output_sents.append(sent[1:])
    output_sents = torch.FloatTensor(output_sents)
    input_sents = torch.FloatTensor(input_sents)
    return input_sents, output_sents

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # RNN Layer
        self.soft_max = nn.Softmax(dim=1)
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = self.soft_max(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

def predict(model, word):
    model.eval()
    out, hidden = model(word)
    return out, hidden

def sample(model, out_len, start):
    model.eval() # eval mode
    
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    sent_vects = list()
    sent_vects.append(list())
    starts = word_tokenize(start.lower())
    for word in starts:
        if word in word_to_vect.keys():
            sent_vects[0].append(word_to_vect[word])
        else:
            sent_vects[0].append([0]*dict_size)
    sent_vects = torch.FloatTensor(sent_vects)
    size = out_len - 1
    # Now pass in the previous characters and get a new one
    for i in range(size):
        word, h = predict(model, sent_vects)
        maxed = [0] * dict_size
        maxed[torch.argmax(word[-1]).item()] = 1
        maxed = torch.FloatTensor(maxed)
        reshaped_word = torch.reshape(maxed, (1, 1, dict_size)) 
        
        sent_vects = torch.cat((sent_vects, reshaped_word), 1)
    out_sent = list()
    for word in sent_vects[0]:
        out_sent.append(bible_words[torch.argmax(word).item()])
    return out_sent



#initialize dataset
num_sents = 100
gut_sents = gutenberg.sents('bible-kjv.txt')[:num_sents]

#process dataset into usable form
bible_sents = preprocess_sents(gut_sents)

#key metadata
print(bible_sents)
bible_words = word_list(bible_sents)
dict_size = len(bible_words)
max_len = len(max(bible_sents, key=len))

#my choice of embedding
#dictionary to read embeddings later
word_to_vect = dict()
for i in range(dict_size):
    word_to_vect[bible_words[i]] = [0] * dict_size
    word_to_vect[bible_words[i]][i] = 1
#transform into input and output
#specific form from tutorial
#https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
vect_sents = vectorize_sents(bible_sents, word_to_vect, dict_size, max_len)
input_sents, output_sents = input_output(vect_sents)
print('Preprocessing Complete\n\n')



#super parameters
hidden_dim = 100
n_layers = 1
model = Model(dict_size, dict_size, hidden_dim, n_layers)
n_epochs = 300
lr=.001

#following juratsky textbook
criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr=lr)





print("Training Model\n---------------")
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    sent_i = 0
    output, hidden = model(input_sents)
    loss = criterion(output, output_sents.view(-1, dict_size))
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    if epoch%2 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))

print("Evaluating Model")

while(True):
    
    print("Enter start of sentence: " )
    start = input()
    print("Enter length of sentence: ")
    length = input()
    print(sample(model, int(length), start)) 

