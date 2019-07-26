import json
from tqdm import tqdm
import numpy as np
import os
import time

import torch
from torch import nn, optim
from tensorboard_logger import configure, log_value

from ntm import NTM
from ntm.datasets import CopyDataset, RepeatCopyDataset, AssociativeDataset, NGram, PrioritySort
from args import get_parser
from marnn import *
from dnc import DNC
from dnc.sam import SAM

args = get_parser().parse_args()
print("args:\n",args)

configure("runs/")
print('name:',args.name)

# ----------------------------------------------------------------------------
# -- initialize datasets, model, criterion and optimizer
# ----------------------------------------------------------------------------

'''
'''


if args.task=='copy':
    args.task_json = 'ntm/tasks/copy.json'
    task_params = json.load(open(args.task_json))
    task_params['max_seq_len']=args.max_seq_len
    dataset = CopyDataset(task_params)
    input_size=task_params['seq_width']+2
    output_size=task_params['seq_width']
elif args.task=='repeat':
    args.task_json = 'ntm/tasks/repeatcopy.json'
    task_params = json.load(open(args.task_json))
#    # (Number of repetition generalisation)
    dataset = RepeatCopyDataset(task_params)
    input_size=task_params['seq_width']+2
    output_size=task_params['seq_width']+1
elif args.task=='recall':
    args.task_json = 'ntm/tasks/associative.json'
    task_params = json.load(open(args.task_json))
    task_params['seq_len']=args.seq_len
    task_params['min_item']=2
    task_params['max_item']=args.max_item
    dataset = AssociativeDataset(task_params)
    input_size=task_params['seq_width']+2
    output_size=task_params['seq_width']
elif args.task=='ngram':
    args.task_json = 'ntm/tasks/ngram.json'
    task_params = json.load(open(args.task_json))
    dataset = NGram(task_params)
    input_size=1
    output_size=1
elif args.task=='sort':
    args.task_json = 'ntm/tasks/prioritysort.json'
    task_params = json.load(open(args.task_json))
    task_params['input_seq_len']=args.input_seq_len
    task_params['target_seq_len']=args.target_seq_len
    dataset = PrioritySort(task_params)
    input_size=task_params['seq_width']+1
    output_size=task_params['seq_width']

"""
For the Copy task, input_size: seq_width + 2, output_size: seq_width
For the RepeatCopy task, input_size: seq_width + 2, output_size: seq_width + 1
For the Associative task, input_size: seq_width + 2, output_size: seq_width
For the NGram task, input_size: 1, output_size: 1
For the Priority Sort task, input_size: seq_width + 1, output_size: seq_width
"""
has_tau=0
if args.model=='ntm':
    model = NTM(input_size= input_size,
          output_size=output_size,
          controller_size=args.lstm_size,
          memory_units=128,
          memory_unit_size=20,
          num_heads=1)#task_params['num_heads'])
elif args.model=='dnc':
    model = DNC(input_size= input_size,
          output_size=output_size,
          hidden_size=args.lstm_size,
          nr_cells=128,
          cell_size=20,
          read_heads=1)#task_params['num_heads'])
    model.init_param()
elif args.model=='sam':
    model = SAM(input_size= input_size,
          output_size=output_size,
          hidden_size=args.lstm_size,
          nr_cells=128,
          cell_size=20,
          read_heads=1)#read_heads=4???#task_params['num_heads'])
    model.init_param()
elif args.model=='lstm':
    marnn_config=args
    print('marnn_config:\n',marnn_config)
    model = MARNN(marnn_config,input_size=input_size,
            num_units=marnn_config.lstm_size,
            output_size=output_size,
            use_zoneout=False,
            use_ln=False)
else:
    has_tau=1
    marnn_config=args
    print('marnn_config:\n',marnn_config)
    model = MARNN(marnn_config,input_size=input_size,
            num_units=marnn_config.lstm_size,
            output_size=output_size,
            use_zoneout=False,
            use_ln=False)
params=0
for p in model.parameters():
    params+=p.numel()
print('Number of parameters:',params)

criterion = nn.BCELoss()
# As the learning rate is task specific, the argument can be moved to json file
if args.optim=='rmsp':
    optimizer = optim.RMSprop(model.parameters(),
                              lr=args.lr,
                              alpha=args.alpha,
                              momentum=args.momentum)
#else:
#    optimizer = optim.Adam(model.parameters(), lr=args.lr,eps=1e-5)

#args.saved_model = 'saved_model_copy.pt'
'''
args.saved_model = 'saved_model_repeatcopy.pt'
args.saved_model = 'saved_model_associative.pt'
args.saved_model = 'saved_model_ngram.pt'
args.saved_model = 'saved_model_prioritysort.pt'
'''

PATH = os.path.join('./save', args.name+'_'+args.task+'.pth')

# ----------------------------------------------------------------------------
# -- basic training loop
# ----------------------------------------------------------------------------
losses = []
errors = []
best_loss=1000000.
start_time=time.time()
for iter in range(args.num_iters):
#for iter in tqdm(range(args.num_iters)):
    optimizer.zero_grad()
    model.reset()

    data = dataset[iter]
    input, target = data['input'], data['target']
    out = torch.zeros(target.size())

    # -------------------------------------------------------------------------
    # loop for other tasks
    # -------------------------------------------------------------------------
    for i in range(input.size()[0]):
        # to maintain consistency in dimensions as torch.cat was throwing error
        in_data = torch.unsqueeze(input[i], 0)
        model(in_data)

    # passing zero vector as input while generating target sequence
    in_data = torch.unsqueeze(torch.zeros(input.size()[1]), 0)
    for i in range(target.size()[0]):
         out[i] = model(in_data)
    # -------------------------------------------------------------------------
    # loop for NGram task
    # -------------------------------------------------------------------------
#    for i in range(task_params['seq_len'] - 1):
#        in_data = input[i].view(1, -1)
#        model(in_data)
#        target_data = torch.zeros([1]).view(1, -1)
#        out[i] = model(target_data)
    # -------------------------------------------------------------------------
    loss = criterion(out, target)
    losses.append(loss.item())
    loss.backward()
    # clips gradient in the range [-10,10]. Again there is a slight but
    # insignificant deviation from the paper where they are clipped to (-10,10)
    nn.utils.clip_grad_value_(model.parameters(), 10)
    optimizer.step()

    binary_output = out.clone()
    binary_output = binary_output.detach().apply_(lambda x: 0 if x < 0.5 else 1)

    # sequence prediction error is calculted in bits per sequence
    error = torch.sum(torch.abs(binary_output - target))
    errors.append(error.item())

    # ---logging---
    if iter % 200 == 0:
        if (iter%400==0) and has_tau:
            if model.cell.tau<marnn_config.max_tau:
                model.cell.set_tau(model.cell.tau+1.)
                print('=======>set tau:',model.cell.tau)
        sec=time.time()-start_time
        min=sec//60
        sec=sec%60
        print('Iteration: %d\tLoss: %.4f\tError in bits per sequence: %.4f, time elapsed:%dmin%.2fsec' %
              (iter, np.mean(losses), np.mean(errors),min,sec))
        print(iter,np.mean(losses),np.mean(errors))
        log_value('train_loss', np.mean(losses), iter)
        log_value('bit_error_per_sequence', np.mean(errors), iter)
        if best_loss>=np.mean(losses):
            best_loss=np.mean(losses)
            best_state_dict=model.state_dict()
            torch.save(best_state_dict, PATH+'.best')
        losses = []
        errors = []
        
torch.save(model.state_dict(),PATH)
# ---saving the model---
