import torch
import helper
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


'''
Custom LSTM in PyTorch as seen in:
https://pdfs.semanticscholar.org/1154/0131eae85b2e11d53df7f1360eeb6476e7f4.pdf
http://www.bioinf.jku.at/publications/older/2604.pdf
'''


class LSTMCell(nn.Module):
    """docstring for LN_LSTMCell"""
    def __init__(self,config,input_size,num_units, 
                 f_bias = 1.):
        super().__init__()

        self.input_size=input_size
        self.num_units = num_units
        self.f_bias = f_bias

        self.use_ln=config.use_ln
        self.use_zoneout  = config.use_zoneout

        x_size=input_size
        h_size=num_units
        self.W_full = nn.Parameter(helper.orthogonal_initializer([x_size+h_size, 4 * h_size] , scale = 1.0))
        self.bias = nn.Parameter(torch.zeros([4*h_size]))

        self.ln1=helper.layernorm(num_units,4)
        self.ln2=helper.layernorm(num_units,1)
        self.zoneout=helper.zoneout(config.zoneout_h,config.zoneout_c)

    def forward(self, x):
        h, c = self.states
        h_size = self.num_units
        x_size = self.input_size
        
        concat = torch.cat((x,h), dim= 1)

        concat = torch.mm(concat,self.W_full) + self.bias
        if self.use_ln:
            concat = self.ln1(concat)

        i,j,f,o = torch.split(concat,h_size, dim=1)

        new_c = c * torch.sigmoid(f + self.f_bias) + torch.sigmoid(i) * torch.tanh(j)
        if self.use_ln:
            new_h = torch.tanh(self.ln2(new_c)) * torch.sigmoid(o)
        else:
            new_h = torch.tanh(new_c) * torch.sigmoid(o)

        if self.use_zoneout:
            new_h, new_c = self.zoneout(new_h, new_c, h, c)

        self.states=(new_h,new_c)
        return new_h

    def reset(self, batch_size=1):
            self.states = (torch.zeros(batch_size, self.num_units).cuda().double(), torch.zeros(batch_size, self.num_units).cuda().double())

#class LSTMCell(nn.Module):
#    def __init__(self, input_size,
#                    hidden_size,
#                    weight_init=None,
#                    reccurent_weight_init=None,
#                    drop=None,
#                    rec_drop=None):
#        super(LSTMCell, self).__init__()
#
#        print("Initializing LSTMCell")
#        self.hidden_size = hidden_size
#        if(weight_init==None):
#            self.W_f = nn.Parameter(torch.zeros(input_size, hidden_size))
#            self.W_f = nn.init.xavier_normal_(self.W_f)
#            self.W_i = nn.Parameter(torch.zeros(input_size, hidden_size))
#            self.W_i = nn.init.xavier_normal_(self.W_i)
#            self.W_o = nn.Parameter(torch.zeros(input_size, hidden_size))
#            self.W_o = nn.init.xavier_normal_(self.W_o)
#            self.W_c = nn.Parameter(torch.zeros(input_size, hidden_size))
#            self.W_c = nn.init.xavier_normal_(self.W_c)
#        else:
#            self.W_f = nn.Parameter(torch.zeros(input_size, hidden_size))
#            self.W_f = weight_init(self.W_f)
#            self.W_i = nn.Parameter(torch.zeros(input_size, hidden_size))
#            self.W_i = weight_init(self.W_i)
#            self.W_o = nn.Parameter(torch.zeros(input_size, hidden_size))
#            self.W_o = weight_init(self.W_o)
#            self.W_c = nn.Parameter(torch.zeros(input_size, hidden_size))
#            self.W_c = weight_init(self.W_c)
#
#        if(reccurent_weight_init == None):
#            self.U_f = nn.Parameter(torch.zeros(hidden_size, hidden_size))
#            self.U_f = nn.init.orthogonal_(self.U_f)
#            self.U_i = nn.Parameter(torch.zeros(hidden_size, hidden_size))
#            self.U_i = nn.init.orthogonal_(self.U_i)
#            self.U_o = nn.Parameter(torch.zeros(hidden_size, hidden_size))
#            self.U_o = nn.init.orthogonal_(self.U_o)
#            self.U_c = nn.Parameter(torch.zeros(hidden_size, hidden_size))
#            self.U_c = nn.init.orthogonal_(self.U_c)
#        else:
#            self.U_f = nn.Parameter(torch.zeros(hidden_size, hidden_size))
#            self.U_f = recurrent_weight_initializer(self.U_f)
#            self.U_i = nn.Parameter(torch.zeros(hidden_size, hidden_size))
#            self.U_i = recurrent_weight_initializer(self.U_i)
#            self.U_o = nn.Parameter(torch.zeros(hidden_size, hidden_size))
#            self.U_o = recurrent_weight_initializer(self.U_o)
#            self.U_c = nn.Parameter(torch.zeros(hidden_size, hidden_size))
#            self.U_c = recurrent_weight_initializer(self.U_c)
#
#        self.b_f = nn.Parameter(torch.zeros(hidden_size))
#        self.b_i = nn.Parameter(torch.zeros(hidden_size))
#        self.b_o = nn.Parameter(torch.zeros(hidden_size))
#        self.b_c = nn.Parameter(torch.zeros(hidden_size))
#
#        if(drop==None):
#            self.keep_prob = False
#        else:
#            self.keep_prob = True
#            self.dropout = nn.Dropout(drop)
#        if(rec_drop == None):
#            self.rec_keep_prob = False
#        else:
#            self.rec_keep_prob = True
#            self.rec_dropout = nn.Dropout(rec_drop)
#
#        self.states = None
#
#    def reset(self, batch_size=1, cuda=True):
#        if cuda:
#            self.states = (Variable(torch.randn(batch_size, self.hidden_size)).cuda().double(), Variable(torch.randn(batch_size, self.hidden_size)).cuda().double())
#        else:
#            self.states = (Variable(torch.randn(batch_size, self.hidden_size)).double(), Variable(torch.randn(batch_size, self.hidden_size)).double())
#
#    def forward(self, X_t):
#        h_t_previous, c_t_previous = self.states
#
#        if self.keep_prob:
#            X_t = self.dropout(X_t)
#        if self.rec_keep_prob:
#            h_t_previous = self.rec_dropout(h_t_previous)
#            c_t_previous = self.rec_dropout(c_t_previous)
#
#
#        f_t = F.sigmoid(
#            torch.mm(X_t, self.W_f) + torch.mm(h_t_previous, self.U_f) + self.b_f #w_f needs to be the previous input shape by the number of hidden neurons
#        )
#
#
#        i_t = F.sigmoid(
#            torch.mm(X_t, self.W_i) + torch.mm(h_t_previous, self.U_i) + self.b_i
#        )
#
#
#        o_t = F.sigmoid(
#            torch.mm(X_t, self.W_o) + torch.mm(h_t_previous, self.U_o) + self.b_o
#        )
#
#
#        c_hat_t = F.tanh(
#            torch.mm(X_t, self.W_c) + torch.mm(h_t_previous, self.U_c) + self.b_c
#        )
#
#        c_t = (f_t * c_t_previous) + (i_t * c_hat_t)
#
#        h_t = o_t * F.tanh(c_t)
#
#        self.states = (h_t, c_t)
#        return h_t

class LSTM(nn.Module):

    def __init__(self,config,
                 input_size=1,
                 hidden_size=64,
                 output_size=1,
                 layers=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers

        self.lstms = nn.ModuleList()
        self.lstms.append(LSTMCell(config,input_size, hidden_size))
        for i in range(self.layers-1):
            self.lstms.append(LSTMCell(config,hidden_size,hidden_size))
        self.fc1 = nn.Linear(hidden_size, output_size)

        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.constant_(self.fc1.bias.data, 0)

    def reset(self, batch_size=1, cuda=True):
        for i in range(len(self.lstms)):
            self.lstms[i].reset(batch_size=batch_size)

    def forward(self, x):
        for i in range(len(self.lstms)):
            x = self.lstms[i](x)
        o = self.fc1(x)
        return o
