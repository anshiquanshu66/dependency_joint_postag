__author__ = 'Dung Doan'

from torch.autograd import Variable

def assign_tensor(tensor, val):

    if isinstance(tensor, Variable):
        assign_tensor(tensor.data, val)
        return tensor
    return tensor.copy_(val)
