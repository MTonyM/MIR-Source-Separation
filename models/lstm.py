import torch
from torch import nn, optim
from torch.autograd import Variable
from options import *


class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, out_dim):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, 
                            hidden_dim, 
                            n_layer, 
                            batch_first=True)
        self.out_voice = nn.Linear(hidden_dim, out_dim)
        self.out_song = nn.Linear(hidden_dim, out_dim)
        
        
    def forward(self, x, h_state):
        r_out, h_state = self.lstm(x, h_state)

        outs_voice = []
        outs_song = []
        for time_step in range(r_out.size(1)):
            outs_voice.append(self.out_voice(r_out[:, time_step, :]))
            outs_song.append(self.out_song(r_out[:, time_step, :]))
        return torch.cat((torch.stack(outs_voice, dim=1), torch.stack(outs_song, dim=1)), 2), h_state


def get_model(args):
    in_dim = 3 * 513
    hidden_dim = 1000
    n_layer = 3
    out_dim = 513
    model = Rnn(in_dim, hidden_dim, n_layer, out_dim)
    return model
