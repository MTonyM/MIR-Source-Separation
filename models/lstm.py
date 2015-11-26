import torch
from torch import nn, optim
from torch.autograd import Variable
from config import *

class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, out_dim):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.FC = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # h0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        # c0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        out, _ = self.lstm(x)
        
        
#         h, w = out.size(1), out.size(2)
#         out = out.view(-1, h * w)
#         out = self.FC(out)
#         out = out.view(-1, h, w)



        return out
    
def get_model(args):
    in_dim = 513
    hidden_dim= 1026
    n_layer = 3
    out_dim = 2*in_dim
    model = Rnn(in_dim, hidden_dim, n_layer, out_dim)
    return model
