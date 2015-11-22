import torch
from torch import nn, optim
from torch.autograd import Variable
from config import *
import numpy as np

class RNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, out_dim):
        super(RNN, self).__init__()
 
        self.rnn = nn.RNN(  # 这回一个普通的 RNN 就能胜任
            input_size=in_dim,
            hidden_size=hidden_dim,     # rnn hidden unit
            num_layers=n_layer,       # 有几层 RNN layers
            batch_first=True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )
        self.out_voice = nn.Linear(hidden_dim, out_dim)
        self.out_song = nn.Linear(hidden_dim, out_dim)
 
    def forward(self, x, h_state):  # 因为 hidden state 是连续的, 所以我们要一直传递这一个 state
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)
        r_out, h_state = self.rnn(x, h_state)   # h_state 也要作为 RNN 的一个输入
 
        outs_voice = []    # 保存所有时间点的预测值
        outs_song = []    # 保存所有时间点的预测值
        for time_step in range(r_out.size(1)):    # 对每一个时间点计算 output
            outs_voice.append(self.out_voice(r_out[:, time_step, :]))
            outs_song.append(self.out_song(r_out[:, time_step, :]))
        return torch.cat((torch.stack(outs_voice, dim=1), torch.stack(outs_song, dim=1)), 2), h_state


def get_model(args):
    in_dim = 3*513 
    hidden_dim= 1000
    n_layer = 3
    out_dim = 513
    model = RNN(in_dim, hidden_dim, n_layer, out_dim)
    return model
