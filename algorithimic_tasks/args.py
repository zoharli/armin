import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task_json', type=str, default='ntm/tasks/copy.json',
                        help='path to json file with task specific parameters')
    parser.add_argument('-model', default='marnn',type=str)
    parser.add_argument('-task', default='copy',type=str)
    parser.add_argument('-name', default='marnn',type=str)
    parser.add_argument('-saved_model', default='saved_model_copy.pt',
                        help='path to file with final model parameters')
    parser.add_argument('-batch_size', type=int, default=1,
                        help='batch size of input sequence during training')
    parser.add_argument('-num_iters', type=int, default=100000,
                        help='number of iterations for training')

    # todo: only rmsprop optimizer supported yet, support adam too
    parser.add_argument('-optim', type=str, default='rmsp',
                        help='learning rate for rmsprop optimizer')
    parser.add_argument('-lr', type=float, default=1e-4,
                        help='learning rate for rmsprop optimizer')
    parser.add_argument('-momentum', type=float, default=0.9,
                        help='momentum for rmsprop optimizer')
    parser.add_argument('-alpha', type=float, default=0.95,
                        help='alpha for rmsprop optimizer')
    parser.add_argument('-beta1', type=float, default=0.9,
                        help='beta1 constant for adam optimizer')
    parser.add_argument('-beta2', type=float, default=0.999,
                        help='beta2 constant for adam optimizer')

    parser.add_argument('-lstm_size',type=int,default=100)
    parser.add_argument('-r_size',type=int,default=32)
    parser.add_argument('-key_size',type=int,default=5)

    #for copy and repeat copy
    parser.add_argument('-max_seq_len',type=int,default=50)
    #for priority sort 
    parser.add_argument('-input_seq_len',type=int,default=40)
    parser.add_argument('-target_seq_len',type=int,default=30)
    #for associative recall
    parser.add_argument('-seq_len',type=int,default=3)
    parser.add_argument('-min_item',type=int,default=2)
    parser.add_argument('-max_item',type=int,default=6)

    # for MANN
    parser.add_argument('-mem_cap', type=int, default=50, help='')
    parser.add_argument('-step_tau', type=float, default=1, help='')
    parser.add_argument('-max_tau', type=float, default=50, help='')
    parser.add_argument('-max_grad_norm', type=float, default=10., help='')
    return parser
