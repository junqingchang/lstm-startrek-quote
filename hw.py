#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import random
import time
import math
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import numpy as np


# In[21]:


TUTORIAL_DATA = 'data/names/*.txt'
reload_model = 'model10.pt' # If continuing on training, be sure to uncomment model loading below at line 297
criterion = nn.NLLLoss()
learning_rate = 0.0005 # Started 0.0001
num_epoch = 5
print_every = 3000 # 5000
lstm_layers = 2
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
max_length = 50


# In[3]:


def get_all_letters():
    all_letters = string.ascii_letters + "0123456789 .,:!?'[]()/+-="
    return all_letters

def get_data():
    category_lines = {}
    all_categories = ['st']
    category_lines['st']=[]
    filterwords=['NEXTEPISODE']
    with open('./data/star_trek_transcripts_all_episodes.csv', newline='', encoding='UTF-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            for el in row:
                if (el not in filterwords) and (len(el)>1):
#                     print(el)
                    v=el.strip().replace(';','').replace('\"','')#.replace('=','').replace('/',' ').replace('+',' ')
                    category_lines['st'].append(v)

    n_categories = len(all_categories)
    print('done')
    return category_lines,all_categories

def num_all_letters(all_letters):
    n_letters = len(all_letters) + 1 # Plus EOS marker
    return n_letters

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s, all_letters):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename, all_letters):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line, all_letters) for line in lines]

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair(all_categories, category_lines):
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

# One-hot vector for category
def categoryTensor(category, all_categories, n_categories):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line, n_letters, all_letters):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line, all_letters, n_letters):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample(all_categories, category_lines, n_categories, n_letters, all_letters):
    category, line = randomTrainingPair(all_categories, category_lines)
    category_tensor = categoryTensor(category, all_categories, n_categories)
    input_line_tensor = inputTensor(line, n_letters, all_letters)
    target_line_tensor = targetTensor(line, all_letters, n_letters)

    return category_tensor, input_line_tensor, target_line_tensor

def generateTensor(line, n_letters, all_letters):
    input_line_tensor = inputTensor(line, n_letters, all_letters)
    target_line_tensor = targetTensor(line, all_letters, n_letters)
    return input_line_tensor, target_line_tensor

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train(train_data, print_every, rnn, optimizer, device, n_letters, all_letters):
    rnn.train()
    
    iter = 0
    total_loss = 0

    for line in train_data:
        optimizer.zero_grad()
        loss = 0
        iter += 1
        input_line_tensor, target_line_tensor = generateTensor(line, n_letters, all_letters)
        target_line_tensor.unsqueeze_(-1)
        h0 = rnn.initHidden()
        c0 = rnn.initHidden()
        hidden = (h0, c0)
        input_line_tensor, target_line_tensor = input_line_tensor.to(device), target_line_tensor.to(device)

        for i in range(input_line_tensor.size(0)):
            output, hidden = rnn(input_line_tensor[i], hidden)
            l = criterion(output, target_line_tensor[i])
            loss += l
            total_loss += l
            
        loss.backward()
        optimizer.step()
            
        if iter % print_every == 0:
            print()
            print('%s (%d %d%%)' % (timeSince(start), iter, iter / len(train_data) * 100))
            print('Samples:')
            starting_words = ''.join(random.choice(string.ascii_uppercase) for _ in range(15))
            samples(n_letters, all_letters, device, rnn, starting_words)
            rnn.train()

    return total_loss.item()/len(train_data)

def test(test_data, rnn, device):
    rnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        total_loss = 0
        for line in test_data:
            input_line_tensor, target_line_tensor = generateTensor(line, n_letters, all_letters)
            target_line_tensor.unsqueeze_(-1)
            input_line_tensor, target_line_tensor = input_line_tensor.to(device), target_line_tensor.to(device)
            h0 = rnn.initHidden()
            c0 = rnn.initHidden()
            hidden = (h0, c0)
            loss = 0
            for i in range(input_line_tensor.size(0)):
                total += 1
                output, hidden = rnn(input_line_tensor[i], hidden)
                prediction = torch.exp(output)
                m = Categorical(prediction)
                topi = m.sample()[0]
                if int(topi) == int(target_line_tensor[i][0]):
                    correct += 1
                l = criterion(output, target_line_tensor[i])
                loss += l
            total_loss += loss
    return total_loss.item()/len(test_data), correct/total
        


# In[4]:


# Sample from a category and starting letter
def sample(n_letters, all_letters, device, rnn, start_letter='A'):
    rnn.eval()
    with torch.no_grad():  # no need to track history in sampling
        input = inputTensor(start_letter, n_letters, all_letters)
        h0 = rnn.initHidden()
        c0 = rnn.initHidden()
        hidden = (h0, c0)
        
        input = input.to(device)
        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(input[0], hidden, True)
            output = torch.exp(output)
            m = Categorical(output)
            topi = m.sample()[0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter,  n_letters, all_letters).to(device)

        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(n_letters, all_letters, device, rnn, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(n_letters, all_letters, device, rnn, start_letter))


# In[6]:


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, device, temperature):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.n_layers = n_layers
        self.temperature = temperature
        
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers)
        
        self.h2t = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden, sample=False):
        x = input.view(len(input), 1, -1)
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.h2t(lstm_out.view(len(input), -1))
        output = self.dropout(output)
        if sample:
            output = self.softmax(output/self.temperature)
        else:
            output = self.softmax(output)
        return output, hidden
        
    def initHidden(self):
        hidden = torch.zeros(self.n_layers, 1, self.hidden_size).to(self.device)
        return hidden


# In[7]:


all_letters = get_all_letters()
n_letters = num_all_letters(all_letters)

category_lines, all_categories = get_data()

n_categories = len(all_categories)
iter_per_epoch = len(category_lines['st'])

    
X_train, X_test = train_test_split(category_lines['st'], test_size=0.20)

print('# categories:', n_categories, all_categories)
print('# data:', len(category_lines['st']))
print('# train:', len(X_train))
print('# test:', len(X_test))


# In[23]:


rnn = CustomLSTM(n_letters, 200, n_letters, lstm_layers, device, 0.5)

# Restart Training
# rnn.load_state_dict(torch.load(reload_model))

rnn.to(device)
optimizer = optim.SGD(rnn.parameters(), lr=learning_rate, momentum=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# In[ ]:


all_losses = []
test_losses = []
accuracy_ot = []

start = time.time()

for epoch in range(1, num_epoch + 1):
    print('Epoch {} Started'.format(epoch))
    scheduler.step()
    train_loss = train(X_train, print_every, rnn, optimizer, device, n_letters, all_letters)

    all_losses.append(train_loss)
    print()
    print('%s Epoch Train Loss: %.4f' % (timeSince(start), train_loss))
    total_loss = 0
    
    test_loss, accuracy = test(X_test, rnn, device)
    test_losses.append(test_loss)
    accuracy_ot.append(accuracy)
    print('%s Epoch Test Loss: %.4f, Accuracy: %.4f ' % (timeSince(start), test_loss, accuracy*100))
    print('Epoch Samples:')
    starting_words = ''.join(random.choice(string.ascii_uppercase) for _ in range(15))
    samples(n_letters, all_letters, device, rnn, starting_words)
    print()
    
    best_model_wts = rnn.state_dict()
    torch.save(best_model_wts,'./model{}.pt'.format(epoch))
    


# In[31]:


plt.figure()
plt.title('Train Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(all_losses)
plt.show()


# In[32]:


plt.figure()
plt.title('Test Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(test_losses)
plt.show()


# In[33]:


plt.figure()
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(accuracy_ot)
plt.show()





