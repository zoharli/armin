import torch
from torch.utils.data import Dataset
from torch.distributions.binomial import Binomial


class AddDataset(Dataset):
    """A Dataset class to generate random examples for the copy task. Each
    sequence has a random length between `min_seq_len` and `max_seq_len`.
    Each vector in the sequence has a fixed length of `seq_width`. The vectors
    are bounded by start and end delimiter flags.

    To account for the delimiter flags, the input sequence length as well
    width is two more than the target sequence.
    """

    def __init__(self, task_params):
        """Initialize a dataset instance for copy task.
        Arguments
        ---------
        task_params : dict
            A dict containing parameters relevant to copy task.
        """
        self.seq_width = task_params['seq_width']
        self.min_seq_len = 2
        self.max_seq_len = task_params['max_seq_len']

    def __len__(self):
        # sequences are generated randomly so this does not matter
        # set a sufficiently large size for data loader to sample mini-batches
        return 65536

    def __getitem__(self, idx):
        # idx only acts as a counter while generating batches.
        seq_len = torch.randint(
            self.min_seq_len, self.max_seq_len, (1,), dtype=torch.long).item()
        s1 =  torch.rand([seq_len, 1], dtype=torch.float64)
        p2 = torch.randint(
            2, seq_len+1, (1,), dtype=torch.long).item()
        p1 = torch.randint(
            1, p2, (1,), dtype=torch.long).item()
        start=torch.ones(1,1,dtype=torch.float64)*2
        end=torch.ones(1,1,dtype=torch.float64)*3
        s2=torch.cat([torch.zeros(p1-1,1),torch.ones(1,1),torch.zeros(p2-1-p1,1),torch.ones(1,1),torch.zeros(seq_len-p2,1)],dim=0).type(dtype=torch.float64)
        input_seq = torch.cat([s1,start,s2,end],dim=0)
        
        target_seq=torch.ones(1,dtype=torch.float64)*(s1[p1]+s1[p2])
        return {'input': input_seq, 'target': target_seq}
