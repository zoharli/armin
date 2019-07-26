import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import helper

'''
Custom GRU cell and GRU in PyTorch as seen in:
https://arxiv.org/pdf/1406.1078v3.pdf
'''

class MARNN(nn.Module):

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

        self.malstms = nn.ModuleList()
        if config.model_type=='armin':
            cell=ARMIN
        elif config.model_type=='tardis':
            cell=TARDIS
        elif config.model_type=='awta':
            cell=ARMIN_with_TARDIS_addr
        self.malstms.append(cell(config,input_size, hidden_size))
        for i in range(self.layers-1):
            self.malstms.append(cell(config,hidden_size, hidden_size))

        self.fc1 = nn.Linear(config.r_size+hidden_size, output_size)
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.constant_(self.fc1.bias.data, 0)

    def reset(self, batch_size=1, cuda=True):
        for i in range(len(self.malstms)):
            self.malstms[i].reset(batch_size=batch_size, cuda=cuda)
            self.malstms[i]._reset_mem(batch_size=batch_size)

    def forward(self, x):
        for i in range(len(self.malstms)):
            x = self.malstms[i](x)
        o = self.fc1(x)
        return o

class ARMIN(nn.Module):
    def __init__(self,config,input_size,num_units, 
                   f_bias = 1.):
        super().__init__()

        self.input_size=input_size
        self.num_units = num_units
        self.f_bias = f_bias

        self.use_zoneout  = config.use_zoneout
        self.zoneout_keep_h = config.zoneout_h
        self.zoneout_keep_c = config.zoneout_c
        self.use_ln=config.use_ln
        self.use_head=config.use_head

        x_size=input_size
        h_size=num_units
        self.r_size=r_size=config.r_size
            
        self.W_full = nn.Parameter(helper.orthogonal_initializer([x_size+r_size+h_size, r_size+4 * h_size] , scale = 1.0))
        self.bias = nn.Parameter(torch.zeros([r_size+4*h_size]))
        self.W_full1 = nn.Parameter(helper.orthogonal_initializer([x_size+r_size+h_size,  r_size+h_size] , scale = 1.0))
        self.bias1 = nn.Parameter(torch.zeros([r_size+h_size]))

        self.ln1=helper.layernorm(num_units,5)
        self.ln2=helper.layernorm(num_units,1)
        self.ln3=helper.layernorm(num_units,2)
        self.drop=nn.Dropout(p=1-config.keep_prob)
        self.zoneout=helper.zoneout1(config.zoneout_c)

        self.c_bias=nn.Parameter(torch.randn(1,num_units))
        self.hmem_bias=nn.Parameter(torch.zeros(1,config.mem_cap,r_size))
        torch.nn.init.normal_(self.c_bias)

        #self.time_fac=torch.cat([torch.ones(config.head_size),torch.Tensor([config.time_fac])])
        self.memcnt=0
        self.mem_cap=config.mem_cap
        self.tau=0.
        self.fc=nn.Linear(x_size+h_size,config.mem_cap)
        self.fc.weight.data=helper.orthogonal_initializer([self.fc.weight.shape[1],self.fc.weight.shape[0]]).t_()
        torch.nn.init.constant_(self.fc.bias,0.)
        self.trans=nn.Linear(h_size,config.r_size)
        self.trans.weight.data=helper.orthogonal_initializer([self.trans.weight.shape[1],self.trans.weight.shape[0]]).t_()
        torch.nn.init.constant_(self.trans.bias,0.)

    def forward(self,x):
        #if self.indrop:
        #    x=self.drop(x0)
        #else:
        c=self.recurrent_state
        h_size = self.num_units
        x_size = self.input_size
        
        h_read_head=torch.cat([x,c],dim=1)
        h_read_head=self.fc(h_read_head)
        #h_read_head,mem_gate=torch.split(h_read_head,[self.mem_cap,1],1)
        h_entry,h_read_index=self.read(h_read_head,self.tau)
        #h_entry=h_entry*self.gumbel_sigmoid(mem_gate,self.tau)

        new_c=torch.cat([c,h_entry],dim=1)
        concat = torch.cat((x,new_c), dim= 1)
        
        concat1=torch.mm(concat,self.W_full1) +self.bias1
        #concat1 = self.ln3(concat1)
        concat1=torch.sigmoid(concat1)
        concat1=torch.cat([torch.ones_like(x),concat1],dim=1)

        concat = concat*concat1
        concat = torch.mm(concat,self.W_full) + self.bias
##        concat = self.ln1(concat)

        i,j,f,o,om = torch.split(concat,h_size, dim=1)

        new_c = c * torch.sigmoid(f + self.f_bias) + torch.sigmoid(i) * torch.tanh(j)
#        new_c=self.ln2(new_c)
        new_c=torch.tanh(new_c)
        
        new_h = new_c * torch.sigmoid(o)
        r = h_entry * torch.sigmoid(om)

        if self.use_zoneout:
            new_c = self.zoneout(new_c, c)

        if self.memcnt<self.mem_cap:
            h_write_index=torch.cat([torch.zeros(self.memcnt),torch.ones(1),torch.zeros(self.mem_cap-1-self.memcnt)]).unsqueeze(0).cuda().double()
            self.memcnt+=1
        else:
            h_write_index=h_read_index.detach()
        self.write(self.trans(new_c),h_write_index)
        
        new_r=torch.cat([new_h,r],dim=1)
        self.recurrent_state=new_c
        return new_r

    def _reset_mem(self,batch_size):
        self.hmem=torch.zeros(batch_size,self.mem_cap,self.r_size).cuda().double()+self.hmem_bias
        self.memcnt=0

    def _reload_mem(self):
        self.hmem=self.hmem.detach()
        
    def _load_mem_state(self,mem_state):
        self.hmem=mem_state.detach()

    def _get_mem_state(self):
        return self.hmem.detach()

    def set_tau(self,num):
        self.tau=num

    def get_tau(self):
        return self.tau

    def write(self,h,h_index):
        #i_ones=i_index.unsqueeze(2)
        h_ones=h_index.unsqueeze(2)
        #self.imem=i.unsqueeze(1)*i_ones+self.imem*(1.-i_ones)
        self.hmem=h.unsqueeze(1)*h_ones+self.hmem*(1.-h_ones)

    def read(self,h_read_head,tau):
        #i_index,_=self.gumbel_softmax(i_read_head,tau)
        h_index,_=self.gumbel_softmax(h_read_head,tau)
        #i_entry=i_index.unsqueeze(2)*self.imem
        h_entry=h_index.unsqueeze(2)*self.hmem
        #i_entry=i_entry.sum(dim=1)
        h_entry=h_entry.sum(dim=1)
        return h_entry,h_index

    def gumbel_softmax(self,input, tau):
            gumbel = -torch.log(1e-20-torch.log(1e-20+torch.rand(*input.shape))).cuda().double()
            y=torch.nn.functional.softmax((input+gumbel)*tau,dim=1)
            ymax,pos=y.max(dim=1)
            hard_y=torch.eq(y,ymax.unsqueeze(1)).double()
            y=(hard_y-y).detach()+y
            return y,pos


    def reset(self, batch_size=1, cuda=True):
        if cuda:
            self.recurrent_state =torch.tanh(self.c_bias.expand(batch_size,-1)).cuda().double()
        else:
            self.recurrent_state =torch.tanh(self.c_bias.expand(batch_size,-1)).double()


class ARMIN_with_TARDIS_addr(nn.Module):
    def __init__(self,config,input_size,num_units, use_zoneout=True,
                  use_ln=True,indrop=True,outdrop=True, f_bias = 1.):
        super().__init__()

        self.input_size=input_size
        self.num_units = num_units
        self.f_bias = f_bias

        self.use_zoneout  = config.use_zoneout
        self.zoneout_keep_h = config.zoneout_h
        self.zoneout_keep_c = config.zoneout_c
        self.use_ln=config.use_ln

        x_size=input_size
        h_size=num_units
            
        self.W_full = nn.Parameter(helper.orthogonal_initializer([x_size+2*h_size, 5 * h_size] , scale = 1.0))
        self.bias = nn.Parameter(torch.zeros([5*h_size]))
        self.W_full1 = nn.Parameter(helper.orthogonal_initializer([x_size+2*h_size,  2*h_size] , scale = 1.0))
        self.bias1 = nn.Parameter(torch.zeros([2*h_size]))

        self.ln1=helper.layernorm(num_units,5)
        self.ln2=helper.layernorm(num_units,1)
        self.ln3=helper.layernorm(num_units,2)
        self.ln4=helper.layernorm(config.mem_cap,1)
        self.drop=nn.Dropout(p=1-config.keep_prob)
        self.zoneout=helper.zoneout1(config.zoneout_c)

        self.memcnt=0
        self.mem_cap=config.mem_cap
        self.tau=1.

        self.keys= nn.Parameter(torch.zeros(config.mem_cap,config.key_size))
        self.vec_a=nn.Parameter(torch.zeros(h_size//4,1))
        nn.init.orthogonal_(self.keys)
        nn.init.orthogonal_(self.vec_a)
        self.fc=nn.Linear(x_size+2*h_size+config.key_size+config.mem_cap,h_size//4)
        self.fc.weight.data=helper.orthogonal_initializer([self.fc.weight.shape[1],self.fc.weight.shape[0]]).t_()
        torch.nn.init.constant_(self.fc.bias,0)
        self.u_t=None
        self.prev_read_location=None

    def forward(self, x0):
        c=self.recurrent_state
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
        #h_read_head=torch.nn.functional.normalize(h_read_head,dim=1)
        h_read_head=self.ln4(h_read_head)
        h_entry,h_read_index=self.read(h_read_head,self.tau)
        self.prev_read_location=h_read_index
        self.u_t=self.u_t+h_read_index
        new_c=torch.cat([c,h_entry],dim=1)
        concat = torch.cat((x,new_c), dim= 1)

        concat1=torch.mm(concat,self.W_full1) +self.bias1
        if self.use_ln:
            concat1 = self.ln3(concat1)
        concat1=torch.sigmoid(concat1)
        concat1=torch.cat([torch.ones_like(x),concat1],dim=1)

        concat = concat*concat1
        concat = torch.mm(concat,self.W_full) + self.bias
        if self.use_ln:
            concat = self.ln1(concat)

        i,j,f,o,om = torch.split(concat,h_size, dim=1)
        #f=torch.sigmoid(f+self.f_bias)
        #new_c = c * f + (1-f) * torch.tanh(j)

        new_c = c * torch.sigmoid(f + self.f_bias) + torch.sigmoid(i) * torch.tanh(j)
        if self.use_ln:
            new_c=self.ln2(new_c)
        new_h = torch.tanh(new_c) * torch.sigmoid(o)
        r = torch.tanh(h_entry) * torch.sigmoid(om)

        if self.use_zoneout:
            new_c = self.zoneout(new_c, c)

        if self.memcnt<self.mem_cap:
            h_write_index=torch.cat([torch.zeros(self.memcnt),torch.ones(1),torch.zeros(self.mem_cap-1-self.memcnt)]).unsqueeze(0).cuda().double()
            self.memcnt+=1
        else:
            h_write_index=h_read_index.detach()
        self.write(new_c,h_write_index)
        
        new_r=torch.cat([new_h,r],dim=1)
        self.recurrent_state=new_c
        return new_r

    def _reset_mem(self,batch_size):
        self.memcnt=0
        self.hmem=torch.zeros(batch_size,self.mem_cap,self.num_units).cuda().double()
        self.prev_read_location=torch.zeros(batch_size,self.mem_cap).cuda().double()
        self.u_t=torch.zeros(batch_size,self.mem_cap).cuda().double()

    def _reload_mem(self):
        self.hmem=self.hmem.detach()
        self.prev_read_location=self.prev_read_location.detach()
        self.u_t=self.u_t.detach()

    def set_tau(self,num):
        self.tau=num

    def get_tau(self):
        return self.tau

    def write(self,h,h_index):
        h_ones=h_index.unsqueeze(2)
        self.hmem=h.unsqueeze(1)*h_ones+self.hmem*(1.-h_ones)

    def read(self,h_read_head,tau):
        h_index,_=self.gumbel_softmax(h_read_head,tau)
        h_entry=h_index.unsqueeze(2)*self.hmem
        h_entry=h_entry.sum(dim=1)
        return h_entry,h_index

    def gumbel_softmax(self,input, tau):
        gumbel = -torch.log(1e-20-torch.log(1e-20+torch.rand(*input.shape))).cuda().double()
        y=torch.nn.functional.softmax((input+gumbel)*tau,dim=1)
        ymax,pos=y.max(dim=1)
        hard_y=torch.eq(y,ymax.unsqueeze(1)).double()
        y=(hard_y-y).detach()+y
        return y,pos

    def reset(self, batch_size=1, cuda=True):
        if cuda:
            self.recurrent_state= torch.randn(batch_size, self.num_units).cuda().double()
        else:
            self.recurrent_state = torch.randn(batch_size, self.num_units).double()

class TARDIS(nn.Module):
    def __init__(self,config,input_size,num_units, 
                   f_bias = 1.):
        super().__init__()

        self.input_size=input_size
        self.num_units = num_units
        self.f_bias = f_bias

        self.use_zoneout  = config.use_zoneout
        self.zoneout_keep_h=config.zoneout_h
        self.zoneout_keep_c=config.zoneout_c

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
        self.keys= nn.Parameter(torch.zeros(config.mem_cap,config.key_size))
        self.vec_a=nn.Parameter(torch.zeros(h_size//4,1))
        nn.init.orthogonal_(self.keys)
        nn.init.orthogonal_(self.vec_a)
        self.fc=nn.Linear(x_size+r_size+h_size+config.key_size+config.mem_cap,h_size//4)
        self.fc.weight.data=helper.orthogonal_initializer([self.fc.weight.shape[1],self.fc.weight.shape[0]]).t_()
        torch.nn.init.constant_(self.fc.bias,0)
        self.fc1=nn.Linear(h_size,r_size)
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
            h_write_index=torch.cat([torch.zeros(self.memcnt),torch.ones(1),torch.zeros(self.mem_cap-1-self.memcnt)]).unsqueeze(0).double().cuda()
            self.memcnt+=1
        else:
            h_write_index=h_read_index.detach()
        self.write(self.fc1(new_h),h_write_index)
        
        output=torch.cat([new_h,r],dim=1)
        self.recurrent_state=(new_h,new_c)
        return output

    def _reset_mem(self,batch_size):
        self.memcnt=0
        self.hmem=torch.zeros(batch_size,self.mem_cap,self.r_size).double().cuda()
        self.prev_read_location=torch.zeros(batch_size,self.mem_cap).double().cuda()
        self.u_t=torch.zeros(batch_size,self.mem_cap).double().cuda()

    def _reload_mem(self):
        self.hmem=self.hmem.detach()
        self.prev_read_location=self.prev_read_location.detach()
        self.u_t=self.u_t.detach()

    def set_tau(self,num):
        self.tau=num

    def get_tau(self):
        return self.tau

    def write(self,h,h_index):
        h_ones=h_index.unsqueeze(2)
        self.hmem=h.unsqueeze(1)*h_ones+self.hmem*(1.-h_ones)

    def read(self,h_read_head,tau):
        h_index,_=self.gumbel_softmax(h_read_head,tau)
        h_entry=h_index.unsqueeze(2)*self.hmem
        h_entry=h_entry.sum(dim=1)
        return h_entry,h_index

    def gumbel_softmax(self,input, tau):
        gumbel = -torch.log(1e-20-torch.log(1e-20+torch.rand(*input.shape))).double().cuda()
        y=torch.nn.functional.softmax((input+gumbel)*tau,dim=1)
        ymax,pos=y.max(dim=1)
        hard_y=torch.eq(y,ymax.unsqueeze(1)).double()
        y=(hard_y-y).detach()+y
        return y,pos

    def gumbel_sigmoid(self,input,tau):
        gumbel = -torch.log(1e-20-torch.log(1e-20+torch.rand(*input.shape))).double().cuda()
        y=torch.sigmoid((input+gumbel)*tau)
        return y

    def reset(self, batch_size=1, cuda=True):
        if cuda:
            c= torch.randn(batch_size, self.num_units).cuda().double()
            h= torch.randn(batch_size, self.num_units).cuda().double()
        else:
            c= torch.randn(batch_size, self.num_units).double()
            h= torch.randn(batch_size, self.num_units).double()
        self.recurrent_state=(h+torch.tanh(self.h_bias),c+torch.tanh(self.c_bias))

