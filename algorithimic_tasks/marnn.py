from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import random

import helper

class LSTMCell(nn.Module):

    def __init__(self,config,input_size,num_units, use_ln,use_zoneout, f_bias = 1.):
        super().__init__()

        self.input_size=input_size
        self.num_units = num_units
        self.f_bias = f_bias

        self.use_ln=use_ln
        self.use_zoneout  = use_zoneout

        x_size=input_size
        h_size=num_units
        self.W_full = nn.Parameter(helper.orthogonal_initializer([x_size+h_size, 4 * h_size] , scale = 1.0))
        self.bias = nn.Parameter(torch.zeros([4*h_size]))


    def forward(self, x ):
        h, c = self.recurrent_state
        h_size = self.num_units
        x_size = self.input_size
        
        concat = torch.cat((x,h), dim= 1)

        concat = torch.mm(concat,self.W_full) + self.bias

        i,j,f,o = torch.split(concat,h_size, dim=1)

        new_c = c * torch.sigmoid(f + self.f_bias) + torch.sigmoid(i) * torch.tanh(j)
        new_h = torch.tanh(new_c) * torch.sigmoid(o)

        self.recurrent_state=(new_h,new_c)
        return new_h

    def zero_state(self, batch_size):
        h = torch.zeros([batch_size, self.num_units])
        c = torch.zeros([batch_size, self.num_units])
        self.recurrent_state=(h, c)

class MARNN(nn.Module):
    def __init__(self,config,input_size,num_units,output_size, use_zoneout=True,
                  use_ln=True):
        super().__init__()

        if config.model=='armin':
            self.cell=ARMIN(config,input_size,num_units,use_zoneout=use_zoneout,use_ln=use_ln)
        elif config.model=='tardis':
            self.cell=TARDIS(config,input_size,num_units,use_zoneout=use_zoneout,use_ln=use_ln)
        elif config.model=='awta':
            self.cell=ARMIN_with_TARDIS_addr(config,input_size,num_units,use_zoneout=use_zoneout,use_ln=use_ln)
        elif config.model=='lstm':
            self.cell=LSTMCell(config,input_size,num_units,use_zoneout=use_zoneout,use_ln=use_ln)

        if config.model=='lstm':
            self.fc=nn.Linear(num_units,output_size)
        else:
            self.fc=nn.Linear(config.r_size+num_units,output_size)
        self.fc.weight.data=helper.orthogonal_initializer([self.fc.weight.shape[1],self.fc.weight.shape[0]]).t_()
        torch.nn.init.constant_(self.fc.bias,0)

    def forward(self,x):
        return torch.sigmoid(self.fc(self.cell(x)))
        
    def reset(self,batch_size=1):
        self.cell.zero_state(batch_size)
        if hasattr(self.cell,'_reset_mem'):
            self.cell._reset_mem(batch_size)

class ARMIN(nn.Module):
    def __init__(self,config,input_size,num_units, use_zoneout=True,
                  use_ln=True,indrop=True,outdrop=True, f_bias = 1.):
        super().__init__()

        self.input_size=input_size
        self.num_units = num_units
        self.f_bias = f_bias

        self.use_zoneout  = use_zoneout
        self.use_ln=use_ln

        x_size=input_size
        h_size=num_units
        self.r_size=r_size=config.r_size
            
        self.W_full = nn.Parameter(helper.orthogonal_initializer([x_size+r_size+h_size, r_size+4* h_size] , scale = 1.0))
        self.bias = nn.Parameter(torch.zeros([r_size+4*h_size]))
        self.W_full1 = nn.Parameter(helper.orthogonal_initializer([x_size+r_size+h_size,  r_size+h_size] , scale = 1.0))
        self.bias1 = nn.Parameter(torch.zeros([r_size+h_size]))

        self.trans=nn.Linear(x_size+h_size,r_size)

        torch.nn.init.orthogonal_(self.trans.weight)
        torch.nn.init.constant_(self.trans.bias,0)

        self.c_bias=nn.Parameter(torch.rand(1,num_units))
        self.hmem_bias=nn.Parameter(torch.zeros(1,config.mem_cap,r_size))

        self.memcnt=0
        self.mem_cap=config.mem_cap
        self.tau=1.
        self.fc=nn.Linear(x_size+h_size,config.mem_cap)
        self.fc.weight.data=helper.orthogonal_initializer([self.fc.weight.shape[1],self.fc.weight.shape[0]]).t_()
        torch.nn.init.constant_(self.fc.bias,0)

    def forward(self, x0):
        h,c= self.recurrent_state
        x=x0

        h_size = self.num_units
        x_size = self.input_size
        
        h_read_head=torch.cat([x,c],dim=1)
        h_read_head=self.fc(h_read_head)
        h_entry,h_read_index=self.read(h_read_head,self.tau)

        new_c=torch.cat([c,h_entry],dim=1)
        concat = torch.cat((x,new_c), dim= 1)
        
        concat1=torch.mm(concat,self.W_full1) +self.bias1
        concat1=torch.sigmoid(concat1)
        concat1=torch.cat([torch.ones_like(x),concat1],dim=1)

        concat = concat*concat1
        concat = torch.mm(concat,self.W_full) + self.bias

        i,j,f,o,om = torch.split(concat,h_size, dim=1)

        new_c = c * torch.sigmoid(f + self.f_bias) + torch.sigmoid(i) * torch.tanh(j)
        new_c=torch.tanh(new_c) 
        new_h = new_c * torch.sigmoid(o)
        r = h_entry * torch.sigmoid(om)

        if self.memcnt<self.mem_cap:
            h_write_index=torch.cat([torch.zeros(self.memcnt),torch.ones(1),torch.zeros(self.mem_cap-1-self.memcnt)]).unsqueeze(0)
            self.memcnt+=1
        else:
            h_write_index=h_read_index.detach()
        self.write(self.trans(torch.cat([x,new_c],dim=1)),h_write_index)
        
        self.recurrent_state=(new_h,new_c)
        new_r=torch.cat([new_h,r],dim=1)
        return new_r

    def zero_state(self, batch_size):
        h = torch.zeros([batch_size, self.num_units])
        c = torch.zeros([batch_size, self.num_units])+torch.tanh(self.c_bias)
        self.recurrent_state=(h, c)

    def _reset_mem(self,batch_size):
        self.memcnt=0
        self.hmem=torch.zeros(batch_size,self.mem_cap,self.r_size)+self.hmem_bias

    def _reload_mem(self):
        self.hmem=self.hmem.detach()

    def set_tau(self,num):
        self.tau=num

    def write(self,h,h_index):
        h_ones=h_index.unsqueeze(2)
        self.hmem=h.unsqueeze(1)*h_ones+self.hmem*(1.-h_ones)

    def read(self,h_read_head,tau):
        h_index=torch.nn.functional.softmax(h_read_head,dim=1)
        h_entry=h_index.unsqueeze(2)*self.hmem
        h_entry=h_entry.sum(dim=1)
        return h_entry,h_index

    def gumbel_softmax(self,input, tau):
        gumbel = -torch.log(1e-20-torch.log(1e-20+torch.rand(*input.shape)))
        y=torch.nn.functional.softmax((input+gumbel)*tau,dim=1)
        ymax,pos=y.max(dim=1)
        hard_y=torch.eq(y,ymax.unsqueeze(1)).float()
        y=(hard_y-y).detach()+y
        return y,pos

    def _reset_inf_mem(self):
        self.memcnt=0
        self.hmem=[]
        for _ in range(self.mem_cap):
            self.hmem.append(torch.zeros(1,self.num_units).cuda())

class TARDIS(nn.Module):
    def __init__(self,config,input_size,num_units, use_zoneout=True,
                  use_ln=True,indrop=True,outdrop=True, f_bias = 1.):
        super().__init__()

        self.input_size = input_size
        self.num_units = num_units
        self.f_bias = f_bias

        self.use_zoneout  = use_zoneout
        self.use_ln=use_ln
        self.indrop=indrop
        self.outdrop=outdrop

        x_size=input_size
        h_size=num_units
        self.r_size=r_size=config.r_size
            
        self.W_full = nn.Parameter(helper.orthogonal_initializer([x_size+r_size+h_size, 3 * h_size] , scale = 1.0))
        self.bias = nn.Parameter(torch.zeros([3*h_size]))
        self.W_full1 = nn.Parameter(helper.orthogonal_initializer([x_size+r_size+h_size,  2] , scale = 1.0))
        self.bias1 = nn.Parameter(torch.zeros([2]))
        self.W_full2 = nn.Parameter(helper.orthogonal_initializer([x_size+r_size+h_size,  1*h_size] , scale = 1.0))
        self.bias2 = nn.Parameter(torch.zeros([1*h_size]))

        self.memcnt=0
        self.mem_cap=config.mem_cap
        self.tau=1.

        self.c_bias=nn.Parameter(torch.zeros(1,h_size))
        self.h_bias=nn.Parameter(torch.zeros(1,h_size))
        self.hmem_bias=nn.Parameter(torch.zeros(1,config.mem_cap,r_size))
        self.keys= nn.Parameter(torch.zeros(config.mem_cap,config.key_size))
        self.vec_a=nn.Parameter(torch.zeros(h_size//4,1))
        nn.init.orthogonal_(self.keys)
        nn.init.orthogonal_(self.vec_a)
        self.fc=nn.Linear(x_size+r_size+h_size+config.key_size+config.mem_cap,h_size//4)
        self.fc.weight.data=helper.orthogonal_initializer([self.fc.weight.shape[1],self.fc.weight.shape[0]]).t_()
        torch.nn.init.constant_(self.fc.bias,0)
        self.fc1=nn.Linear(h_size+x_size,r_size)
        self.fc1.weight.data=helper.orthogonal_initializer([self.fc1.weight.shape[1],self.fc1.weight.shape[0]]).t_()
        torch.nn.init.constant_(self.fc1.bias,0)
        self.u_t=None
        self.prev_read_location=None

    def forward(self, x0):
        h,c= self.recurrent_state
        x=x0

        h_size = self.num_units
        x_size = self.input_size
        
        h_read_head=torch.cat([torch.cat([x,c],dim=1).unsqueeze(1).expand(-1,self.mem_cap,-1),
                                self.keys.unsqueeze(0).expand(x.shape[0],-1,-1),
                                self.hmem,
                                torch.nn.functional.normalize(self.u_t,dim=1).unsqueeze(1).expand(-1,self.mem_cap,-1)],dim=2)
        h_read_head=self.fc(h_read_head)
        h_read_head=torch.bmm(torch.tanh(h_read_head),self.vec_a.unsqueeze(0).expand(x.shape[0],-1,-1)).squeeze(2)
        h_read_head=h_read_head-self.prev_read_location*100
        #added:
        h_read_head=torch.nn.functional.normalize(h_read_head,dim=1)

        r,h_read_index=self.read(h_read_head,self.tau)
        self.prev_read_location=h_read_index
        self.u_t=self.u_t+h_read_index

        new_h=torch.cat([h,r],dim=1)
        concat0 = torch.cat((x,new_h), dim= 1)
        concat1 = torch.mm(concat0,self.W_full) + self.bias

        i,f,o = torch.split(concat1,h_size, dim=1)

        alpha,beta=torch.split(self.gumbel_sigmoid(torch.mm(concat0,self.W_full1)+self.bias1,10/3),1,dim=1)
        concat2= concat0 
        concat2= concat0 * torch.cat([torch.ones_like(x),torch.ones_like(h)*alpha,torch.ones_like(r)*beta],dim=1)

        new_c=torch.tanh(torch.mm(concat2,self.W_full2)+self.bias2)
        new_c = c * torch.sigmoid(f + self.f_bias) + torch.sigmoid(i) * new_c
        new_h = torch.tanh(new_c) * torch.sigmoid(o)

        if self.memcnt<self.mem_cap:
            h_write_index=torch.cat([torch.zeros(self.memcnt),torch.ones(1),torch.zeros(self.mem_cap-1-self.memcnt)]).unsqueeze(0)
            self.memcnt+=1
        else:
            h_write_index=h_read_index.detach()
        self.write(self.fc1(torch.cat((x,new_h),dim=1)),h_write_index)
        
        output=torch.cat([new_h,r],dim=1)
        self.recurrent_state=(new_h,new_c)
        return output

    def zero_state(self, batch_size):
        h = torch.zeros([batch_size, self.num_units])+torch.tanh(self.h_bias)
        c = torch.zeros([batch_size, self.num_units])+torch.tanh(self.c_bias)
        self.recurrent_state=(h, c)

    def _reset_mem(self,batch_size):
        self.memcnt=0
        self.hmem=torch.zeros(batch_size,self.mem_cap,self.r_size)+self.hmem_bias
        self.prev_read_location=torch.zeros(batch_size,self.mem_cap)
        self.u_t=torch.zeros(batch_size,self.mem_cap)

    def _reload_mem(self):
        self.hmem=self.hmem.detach()
        self.prev_read_location=self.prev_read_location.detach()
        self.u_t=self.u_t.detach()

    def set_tau(self,num):
        self.tau=num

    def write(self,h,h_index):
        h_ones=h_index.unsqueeze(2)
        self.hmem=h.unsqueeze(1)*h_ones+self.hmem*(1.-h_ones)

    def read(self,h_read_head,tau):
        h_index,_=self.gumbel_softmax(h_read_head,tau)
        h_entry=h_index.unsqueeze(2)*self.hmem
        h_entry=h_entry.sum(dim=1)
        return h_entry,h_index

    def gumbel_softmax(self,input, tau):
        gumbel = -torch.log(1e-20-torch.log(1e-20+torch.rand(*input.shape)))
        y=torch.nn.functional.softmax((input+gumbel)*tau,dim=1)
        ymax,pos=y.max(dim=1)
        hard_y=torch.eq(y,ymax.unsqueeze(1)).float()
        y=(hard_y-y).detach()+y
        return y,pos

    def gumbel_sigmoid(self,input,tau):
        gumbel = -torch.log(1e-20-torch.log(1e-20+torch.rand(*input.shape)))
        y=torch.sigmoid((input+gumbel)*tau)
        return y


class ARMIN_with_TARDIS_addr(nn.Module):
    def __init__(self,config,input_size,num_units, use_zoneout=True,
                  use_ln=True,indrop=True,outdrop=True, f_bias = 1.):
        super().__init__()

        self.input_size=input_size
        self.num_units = num_units
        self.f_bias = f_bias

        self.use_zoneout  = use_zoneout
        self.use_ln=use_ln
        self.indrop=indrop
        self.outdrop=outdrop

        x_size=input_size
        h_size=num_units
        self.r_size=r_size=config.r_size
            
        self.W_full = nn.Parameter(helper.orthogonal_initializer([x_size+r_size+h_size, r_size+4* h_size] , scale = 1.0))
        self.bias = nn.Parameter(torch.zeros([r_size+4*h_size]))
        self.W_full1 = nn.Parameter(helper.orthogonal_initializer([x_size+r_size+h_size,  r_size+h_size] , scale = 1.0))
        self.bias1 = nn.Parameter(torch.zeros([r_size+h_size]))

        self.trans=nn.Linear(h_size+x_size,r_size)
        torch.nn.init.orthogonal_(self.trans.weight)
        torch.nn.init.constant_(self.trans.bias,0)

        self.c_bias=nn.Parameter(torch.rand(1,num_units))
        self.hmem_bias=nn.Parameter(torch.zeros(1,config.mem_cap,r_size))

        self.memcnt=0
        self.mem_cap=config.mem_cap
        self.tau=1.

        self.keys= nn.Parameter(torch.zeros(config.mem_cap,config.key_size))
        self.vec_a=nn.Parameter(torch.zeros(h_size//4,1))
        nn.init.orthogonal_(self.keys)
        nn.init.xavier_uniform_(self.vec_a)
        self.fc=nn.Linear(x_size+r_size+h_size+config.key_size+config.mem_cap,h_size//4)
        self.fc.weight.data=helper.orthogonal_initializer([self.fc.weight.shape[1],self.fc.weight.shape[0]]).t_()
        torch.nn.init.constant_(self.fc.bias,0)
        self.u_t=None
        self.prev_read_location=None

    def forward(self, x0,state=None):
        h,c= self.recurrent_state
        x=x0

        h_size = self.num_units
        x_size = self.input_size
        
        h_read_head=torch.cat([torch.cat([x,c],dim=1).unsqueeze(1).expand(-1,self.mem_cap,-1),
                                self.keys.unsqueeze(0).expand(x.shape[0],-1,-1),
                                self.hmem,
                                torch.nn.functional.normalize(self.u_t,dim=1).unsqueeze(1).expand(-1,self.mem_cap,-1)],dim=2)
        h_read_head=self.fc(h_read_head).squeeze(2)
        h_read_head=torch.bmm(torch.tanh(h_read_head),self.vec_a.unsqueeze(0).expand(x.shape[0],-1,-1)).squeeze(2)
        h_read_head=h_read_head-self.prev_read_location*100

        h_read_head=torch.nn.functional.normalize(h_read_head,dim=1)
        h_entry,h_read_index=self.read(h_read_head,self.tau)
        self.prev_read_location=h_read_index
        self.u_t=self.u_t+h_read_index
        new_c=torch.cat([c,h_entry],dim=1)
        concat = torch.cat((x,new_c), dim= 1)

        concat1=torch.mm(concat,self.W_full1) +self.bias1
        concat1=torch.sigmoid(concat1)
        concat1=torch.cat([torch.ones_like(x),concat1],dim=1)

        concat = concat*concat1
        concat = torch.mm(concat,self.W_full) + self.bias

        i,j,f,o,om = torch.split(concat,h_size, dim=1)

        new_c = c * torch.sigmoid(f + self.f_bias) + torch.sigmoid(i) * torch.tanh(j)
        new_c=torch.tanh(new_c)
        new_h = new_c * torch.sigmoid(o)
        r = h_entry * torch.sigmoid(om)

        if self.memcnt<self.mem_cap:
            h_write_index=torch.cat([torch.zeros(self.memcnt),torch.ones(1),torch.zeros(self.mem_cap-1-self.memcnt)]).unsqueeze(0)
            self.memcnt+=1
        else:
            h_write_index=h_read_index.detach()

        self.write(self.trans(torch.cat((x,new_c),dim=1)),h_write_index)
        
        new_r=torch.cat([new_h,r],dim=1)
        self.recurrent_state=(new_h,new_c)
        return new_r

    def zero_state(self, batch_size):
        h = torch.zeros([batch_size, self.num_units])
        c = torch.zeros([batch_size, self.num_units])+torch.tanh(self.c_bias)
        self.recurrent_state=(h,c)

    def _reset_mem(self,batch_size):
        self.memcnt=0
        self.hmem=torch.zeros(batch_size,self.mem_cap,self.r_size)+self.hmem_bias
        self.prev_read_location=torch.zeros(batch_size,self.mem_cap)
        self.u_t=torch.zeros(batch_size,self.mem_cap)

    def _reload_mem(self):
        self.hmem=self.hmem.detach()
        self.prev_read_location=self.prev_read_location.detach()
        self.u_t=self.u_t.detach()

    def set_tau(self,num):
        self.tau=num

    def write(self,h,h_index):
        h_ones=h_index.unsqueeze(2)
        self.hmem=h.unsqueeze(1)*h_ones+self.hmem*(1.-h_ones)

    def read(self,h_read_head,tau):
        h_index,_=self.gumbel_softmax(h_read_head,tau)
        h_entry=h_index.unsqueeze(2)*self.hmem
        h_entry=h_entry.sum(dim=1)
        return h_entry,h_index

    def gumbel_softmax(self,input, tau):
        gumbel = -torch.log(1e-20-torch.log(1e-20+torch.rand(*input.shape)))
        y=torch.nn.functional.softmax((input+gumbel)*tau,dim=1)
        ymax,pos=y.max(dim=1)
        hard_y=torch.eq(y,ymax.unsqueeze(1)).float()
        y=(hard_y-y).detach()+y
        return y,pos

