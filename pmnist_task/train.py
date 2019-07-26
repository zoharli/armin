import argparse
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import configparser
from gpu import *

config = configparser.ConfigParser()

sys.path.append('./task/')

'''Baseline Dataset'''
from sequential_mnist import SequentialMNIST

'''Models of Interest'''
from lstm import LSTM
from marnn import MARNN
from ntm import NTM
from dnc import DNC
from setproctitle import *

parser = argparse.ArgumentParser(description='Nueron Connection')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--optim', type=str, default='adam',
                    help='optim method')
parser.add_argument('--decay', type=str, default='fix',
                    help='decay method')
parser.add_argument('--patient', type=int, default=10)
parser.add_argument('--name', type=str, default='pmnist')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='RMSprop optimizer momentum (default: 0.9)')
parser.add_argument('--alpha', type=float, default=0.95,
                    help='RMSprop alpha (default: 0.95)')
parser.add_argument('--epochs', type=int, default=150,
                    help='num training epochs (default: 100)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--hx', type=int, default=100,
                    help='hidden vec size for lstm models (default: 100)')
parser.add_argument('--r_size', type=int, default=28,
                    help='hidden vec size for lstm models (default: 100)')
parser.add_argument('--layers', type=int, default=1,
                    help='num recurrent layers (default: 1)')
parser.add_argument('--batch-size', type=int, default=128,
                    help='training batch size (default: 64)')
parser.add_argument('--model-type', type=str, default='lstm',
                    help='lstm, gru, irnn, ugrnn, rnn+, peephole')
parser.add_argument('--task', type=str, default='pseqmnist',
                    help='seqmnist, pseqmnist')
parser.add_argument('--sequence-len', type=int, default=784,
                    help='mem seq len (default: 784)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use gpu')
parser.add_argument('--save', type=str, default='save', help='')
parser.add_argument('--mem_cap', type=int, default=28, help='')
parser.add_argument('--use_head', type=int, default=0, help='')
parser.add_argument('--key_size', type=int, default=25, help='')
parser.add_argument('--step_tau', type=float, default=1, help='')
parser.add_argument('--use_zoneout', type=int, default=0, help='')
parser.add_argument('--use_ln', type=int, default=0, help='')
parser.add_argument('--zoneout_c', type=float, default=0.8, help='')
parser.add_argument('--zoneout_h', type=float, default=0.8, help='')
parser.add_argument('--keep_prob', type=float, default=1., help='')
parser.add_argument('--max_grad_norm', type=float, default=1, help='')
parser.add_argument('--eps', type=float, default=1e-8, help='')
parser.add_argument('--gpu', type=str, default='', help='')
parser.add_argument('--init_from', type=str, default='', help='')



args = parser.parse_args()
setproctitle(args.name)
find_idle_gpu(args.gpu)
args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    print("Using GPU Acceleration")

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

np.random.seed(args.seed)
random.seed(args.seed)


def log_sigmoid(x):
    return torch.log(torch.sigmoid(x))

if args.task == 'seqmnist':
    print("Loading SeqMNIST")
    dset = SequentialMNIST()
else:
    print("Loading PSeqMNIST")
    dset = SequentialMNIST(permute=True)

data_loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=True)
activation = nn.LogSoftmax(dim=1)
criterion = nn.CrossEntropyLoss()
#criterion = nn.CrossEntropyLoss(size_average=False, reduce=False)

def create_model():
    if args.model_type == 'lstm':
        return LSTM(config=args,input_size=dset.input_dimension,
                                          hidden_size=args.hx,
                                          output_size=dset.output_dimension,
                                          layers=args.layers)

    elif args.model_type == 'armin' or args.model_type=='tardis' or args.model_type=='awta':
        return MARNN(args,dset.input_dimension,
                                          args.hx,
                                          dset.output_dimension,
                                          )
    elif args.model_type=='ntm':
        return NTM(dset.input_dimension,dset.output_dimension,args.hx,args.mem_cap,args.r_size,1)

    else:
        raise Exception

model = create_model()
model.double()

params = 0
for p in list(model.parameters()):
    params += p.numel()
print ("Num params: ", params)
print (model)
if args.cuda:
    model.cuda()

if args.optim=='rmsp':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
elif args.optim=='adam':
    optimizer = optim.Adam(model.parameters(),lr=args.lr,eps=args.eps)



def run_sequence(seq, target):
    predicted_list = []
    y_list = []
    model.reset(batch_size=seq.size(0), cuda=args.cuda)
    for i, input_t in enumerate(seq.chunk(seq.size(1), dim=1)):
        input_t = input_t.squeeze(1)
        if activation == None:
            p = model(input_t)
        else:
            p = model(input_t)
            p = activation(p)
        predicted_list.append(p)
        y_list.append(target)

    return predicted_list, y_list

def train(epoch):
    model.train()
    dset.train()
    if isinstance(model,MALSTM):
        for cell in model.malstms:
            if cell.get_tau()<args.mem_cap-1:
                if epoch>=args.mem_cap:
                    cell.set_tau(args.mem_cap-1)
                else:
                    cell.set_tau(cell.get_tau()+1)
                print('===========>set tau:',model.malstms[0].get_tau())
    total_loss = 0.0
    steps = 0
    n_correct = 0
    n_possible = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda().double(), target.cuda().double()
        optimizer.zero_grad()
        predicted_list, y_list = run_sequence(data, target)

        pred = predicted_list[-1]
        y_ = y_list[-1].long()
        prediction = pred.data.max(1, keepdim=True)[1].long()
        n_correct += prediction.eq(y_.data.view_as(prediction)).sum().cpu().numpy()
        n_possible += int(prediction.shape[0])
        loss = F.nll_loss(pred, y_)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)
        optimizer.step()
        steps += 1
        total_loss += loss.cpu().data.numpy()
        optimizer.zero_grad()

    print("Train loss ", total_loss/steps)
    print("Train Acc ", (n_correct/ n_possible))

def validate(epoch):
    dset.val()
    model.eval()

    total_loss = 0.0
    n_correct = 0
    n_possible = 0
    steps = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda().double(), target.cuda().double()
        with torch.no_grad():
            data, target = data, target
            predicted_list, y_list = run_sequence(data, target)

        pred = predicted_list[-1]
        y_ = y_list[-1].long()
        prediction = pred.data.max(1, keepdim=True)[1].long() # get the index of the max log-probability
        n_correct += prediction.eq(y_.data.view_as(prediction)).sum().cpu().numpy()
        n_possible += int(prediction.shape[0])
        loss = F.nll_loss(pred, y_)

        steps += 1
        total_loss += loss.cpu().data.numpy()

        optimizer.zero_grad()

    acc=n_correct/n_possible
    print("Validation Acc ", acc)
    return total_loss / steps,acc

def test():
    dset.test()
    model.eval()

    total_loss = 0.0
    n_correct = 0
    n_possible = 0
    steps = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda().double(), target.cuda().double()
        with torch.no_grad():
            data, target = data, target
            predicted_list, y_list = run_sequence(data, target)

        pred = predicted_list[-1]
        y_ = y_list[-1].long()
        prediction = pred.data.max(1, keepdim=True)[1].long() # get the index of the max log-probability
        n_correct += prediction.eq(y_.data.view_as(prediction)).sum().cpu().numpy()
        n_possible += int(prediction.shape[0])
        loss = F.nll_loss(pred, y_)

        steps += 1
        total_loss += loss.cpu().data.numpy()


    acc=n_correct/n_possible
    print("Test Acc ", acc)
    return total_loss / steps,acc

def run():
    for x in args.__dict__:
        print(x,args.__dict__[x])
    best_val_loss=None
    best_acc=None
    best_state_dict=None
    start_epoch=0
    if args.init_from!='':
        state_dict=torch.load(args.init_from)
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optim_state_dict'])
        start_epoch=state_dict['epoch']
        best_val_loss=state_dict['best_val_loss']
        best_acc=state_dict['best_acc']
    if best_val_loss is None:
        best_val_loss = np.inf
    if best_acc is None:
        best_acc=0.
    p=0
    for epoch in range(start_epoch,args.epochs):
        print("\n\n**************************epoch:%d********************************"%epoch)
        tim = time.time()
        train(epoch)
        val_loss,acc = validate(epoch)
        best_acc=max(best_acc,acc)
        print ("Val Loss (epoch", epoch, "): ", val_loss)
        print("best acc :",best_acc )
        print("Epoch time: ", time.time() - tim)
        if args.decay=='fix':
            if epoch==120:
                for pg in optimizer.param_groups:
                    pg['lr']=pg['lr']*0.1
                    lr=pg['lr']   
                print('===========>decay lr:',lr)
        else:
            if p>=args.patient:
                lr=0
                for pg in optimizer.param_groups:
                    pg['lr']=pg['lr']*0.5
                    lr=pg['lr']   
                print('===========>decay lr:',lr)
                p=0
        if val_loss < best_val_loss:
            best_val_loss=val_loss
            best_acc=acc
            best_state_dict=model.state_dict()
            torch.save({'model_state_dict':model.state_dict(),
                        'optim_state_dict':optimizer.state_dict(),
                        'epoch':epoch,
                        'best_acc':best_acc,
                        'best_val_loss':best_val_loss},os.path.join(args.save,args.name+'.pth.best'))
        torch.save({'model_state_dict':model.state_dict(),
                    'optim_state_dict':optimizer.state_dict(),
                    'epoch':epoch,
                    'best_acc':best_acc,
                    'best_val_loss':best_val_loss},os.path.join(args.save,args.name+'.pth'))
    print('**********************************')
    test_loss,acc=test()
    print ("Test Loss : ", test_loss)
    print("acc :",acc )
       

if __name__ == "__main__":
    run()
