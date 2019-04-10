import sys
import unicodedata
import string
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

if len(sys.argv)  != 2:
    print('Usage: python sample.py <starting character/characters>\ne.g. python sample.py A\ne.g. python sample.py KS')
    sys.exit(0)

MODEL = 'model5.pt'
lstm_layers = 2
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
max_length = 100

def get_all_letters():
    all_letters = string.ascii_letters + "0123456789 .,:!?'[]()/+-="
    return all_letters

def num_all_letters(all_letters):
    n_letters = len(all_letters) + 1 # Plus EOS marker
    return n_letters

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line, n_letters, all_letters):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

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

all_letters = get_all_letters()
n_letters = num_all_letters(all_letters)
rnn = CustomLSTM(n_letters, 200, n_letters, lstm_layers, device, 0.4)
rnn.to(device)
rnn.load_state_dict(torch.load(MODEL, map_location=device))
samples(n_letters, all_letters, device, rnn, sys.argv[1])
