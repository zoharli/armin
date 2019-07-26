#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence

from .memory import *
import helper 


from torch.nn.init import orthogonal, xavier_uniform

class LSTMCell(nn.Module):
    """docstring for LSTMCell"""
    def __init__(self,input_size,num_units,  f_bias = 1.):
        super().__init__()

        self.input_size=input_size
        self.num_units = num_units
        self.f_bias = f_bias

        x_size=input_size
        h_size=num_units
        self.W_full = nn.Parameter(helper.orthogonal_initializer([x_size+h_size, 4 * h_size] , scale = 1.0))
        self.bias = nn.Parameter(torch.zeros([4*h_size]))

    def inference(self,x,state):
        return self.forward(x,state)
    
    def forward(self, x ,state):
        h, c = state
        h_size = self.num_units
        x_size = self.input_size
        
        concat = torch.cat((x,h), dim= 1)

        concat = torch.mm(concat,self.W_full) + self.bias

        i,j,f,o = torch.split(concat,h_size, dim=1)

        new_c = c * torch.sigmoid(f + self.f_bias) + torch.sigmoid(i) * torch.tanh(j)
        new_h = torch.tanh(new_c) * torch.sigmoid(o)


        return new_h, new_c

    def zero_state(self, batch_size):
        h = torch.zeros([batch_size, self.num_units])
        c = torch.zeros([batch_size, self.num_units])
        return (h, c)

class DNC(nn.Module):

  def __init__(
      self,
      input_size,
      output_size,
      hidden_size,
      rnn_type='lstm',
      num_layers=1,
      num_hidden_layers=2,
      bias=True,
      batch_first=True,
      dropout=0,
      bidirectional=False,
      nr_cells=5,
      read_heads=2,
      cell_size=10,
      nonlinearity='tanh',
      gpu_id=0,
      independent_linears=True,
      share_memory=True,
      debug=False,
      clip=0
  ):
    super(DNC, self).__init__()
    # todo: separate weights and RNNs for the interface and output vectors

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.rnn_type = rnn_type
    self.num_layers = num_layers
    self.num_hidden_layers = num_hidden_layers
    self.bias = bias
    self.batch_first = batch_first
    self.dropout = dropout
    self.bidirectional = bidirectional
    self.nr_cells = nr_cells
    self.read_heads = read_heads
    self.cell_size = cell_size
    self.nonlinearity = nonlinearity
    self.gpu_id = gpu_id
    self.independent_linears = independent_linears
    self.share_memory = share_memory
    self.debug = debug
    self.clip = clip

    self.w = self.cell_size
    self.r = self.read_heads

    self.read_vectors_size = self.r * self.w

    self.nn_input_size = self.input_size + self.read_vectors_size
    self.nn_output_size = self.hidden_size+ self.read_vectors_size

    self.h_bias=nn.Parameter(torch.zeros(self.hidden_size))
    self.c_bias=nn.Parameter(torch.zeros(self.hidden_size))
    self.last_read_bias=nn.Parameter(torch.zeros(self.w*self.r))

    '''
    self.rnns = []
    self.memories = []
    '''

    if self.rnn_type.lower() == 'lstm':
      self.rnns=LSTMCell(self.nn_input_size,
                                 self.hidden_size)
      '''
      setattr(self, self.rnn_type.lower() + '_layer_' + str(layer), self.rnns[layer])
      '''

    if self.share_memory:
      self.memories=Memory(
              input_size=self.hidden_size,
              mem_size=self.nr_cells,
              cell_size=self.w,
              read_heads=self.r,
              gpu_id=self.gpu_id,
              independent_linears=self.independent_linears
          )
      setattr(self, 'rnn_layer_memory_shared', self.memories)
    # final output layer
    self.output = nn.Linear(self.nn_output_size, self.output_size)
    nn.init.kaiming_uniform_(self.output.weight)

    '''
    if self.gpu_id != -1:
      [x.cuda() for x in self.rnns]
      [x.cuda() for x in self.memories]
    '''

  def init_param(self):
    for n,p in self.named_parameters():
        if 'weight' == n[-6:]:
            torch.nn.init.xavier_normal_(p)
        if 'bias' == n[-4:]:
            torch.nn.init.constant_(p,0)

  def _init_hidden(self, hx, batch_size, reset_experience):
    # create empty hidden states if not provided
    if hx is None:
      hx = (None, None, None)
    (chx, mhx, last_read) = hx

    # initialize hidden state of the controller RNN
    if chx is None:
      h = T.zeros(batch_size, self.hidden_size)

      chx = (h+torch.tanh(self.h_bias), h+torch.tanh(self.c_bias)) if self.rnn_type.lower() == 'lstm' else h

    # Last read vectors
    if last_read is None:
      last_read = T.zeros(batch_size, self.w * self.r)+self.last_read_bias

    # memory states
    if mhx is None:
      if self.share_memory:
        mhx = self.memories.reset(batch_size, erase=reset_experience)
    else:
      if self.share_memory:
        mhx = self.memories.reset(batch_size, mhx, erase=reset_experience)

    return chx, mhx, last_read

  def reset(self,batch_size=1):
      self.recurrent_state=self._init_hidden(None,batch_size,True)

  def zero_state(self,batch_size):
    return self._init_hidden(None,batch_size,False)

  def _reset_mem(self,batch_size):
    mhx = self.memories.reset(batch_size, erase=reset_experience)
      

  def _layer_forward(self, input,  hx=(None, None)) :
    (chx, mhx) = hx
    # pass through the controller layer
    chx = self.rnns(input, chx)
    input = chx[0]
    # clip the controller output
    if self.clip != 0:
      output = T.clamp(input, -self.clip, self.clip)
    else:
      output = input
    # the interface vector
    ξ = output
    # pass through memory
    if self.share_memory:
      read_vecs, mhx = self.memories(ξ, mhx)
  # the read vectors
    read_vectors = read_vecs.view(-1, self.w * self.r)

    return output, (chx, mhx, read_vectors)

  def forward(self, input):

    controller_hidden, mem_hidden, last_read = self.recurrent_state
    # concat input with last read (or padding) vectors
    inputs = T.cat([input, last_read], 1)

    chx = controller_hidden
    m = mem_hidden
    # pass through controller
    outs, (chx, m, read_vectors) = \
      self._layer_forward(inputs,  (chx, m))

    mem_hidden = m
    controller_hidden = chx

  # the controller output + read vectors go into next layer
    outs = T.cat([outs, read_vectors], 1)

    self.recurrent_state= (controller_hidden, mem_hidden, read_vectors)

    return torch.nn.functional.sigmoid(self.output(outs))
