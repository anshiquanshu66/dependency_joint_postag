__author__ = 'Dung Doan'

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from functions import variational_rnn as rnn_F

class VarMaskedRNNBase(nn.Module):
    def __init__(self, Cell, input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
                 dropout=(0,0), bidirectional=False, initializer=None, **kwargs):

        super(VarMaskedRNNBase, self).__init__()
        self.Cell = Cell
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.lstm = False
        num_directions = 2 if bidirectional else 1

        self.all_cells = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                cell = self.Cell(layer_input_size, hidden_size, self.bias, p=dropout, initializer=initializer, **kwargs)
                self.all_cells.append(cell)
                self.add_module('cell%d' % (layer * num_directions + direction), cell)

    def reset_parameters(self):
        for cell in self.all_cells:
            cell.reset_parameters()

    def reset_noise(self, batch_size):
        for cell in self.all_cells:
            cell.reset_noise(batch_size)

    def forward(self, input, mask=None, hx=None):
        batch_size = input.size(0) if self.batch_first else input.size(1)
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.autograd.Variable(input.data.new(self.num_layers * num_directions, batch_size, self.hidden_size).zero_())
            if self.lstm:
                hx = (hx, hx)
        func = rnn_F.AutogradVarMaskedRNN(num_layers=self.num_layers,
                                          batch_first=self.batch_first,
                                          bidirectional=self.bidirectional,
                                          lstm=self.lstm)
        self.reset_noise(batch_size)

        output, hidden = func(input, self.all_cells, hx, None if mask is None else mask.view(mask.size() + (1,)))
        return output, hidden

    def step(self, input, hx=None, mask=None):
        assert not self.bidirectional, "step only cannot be applied to bidirectional RNN."
        batch_size = input.size(0)
        if hx is None:
            hx = torch.autograd.Variable(input.data.new(self.num_layers, batch_size, self.hidden_size).zero_())
            if self.lstm:
                hx = (hx, hx)

        func = rnn_F.AutogradVarMaskedStep(num_layers=self.num_layers, lstm=self.lstm)

        output, hidden = func(input, self.all_cells, hx, mask)
        return output, hidden


class VarRNNCellBase(nn.Module):
    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinerity != "tanh":
            s += ', nonlinearity={nolinearity}'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def reset_noise(self, batch_size):
        raise NotImplementedError

def default_initializer(hidden_size):
    stdv = 1.0 / math.sqrt(hidden_size)
    def forward(tensor):
        nn.init.uniform_(tensor, -stdv, stdv)

    return forward


class VarRNNCell(VarRNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh", p=(0.5, 0.5), initializer=None):
        super(VarRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(hidden_size))
            self.bias_hh = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_id', None)
            self.register_parameter('bias_hh', None)

        self.initializer = default_initializer(self.hidden_size) if initializer is None else initializer
        self.reset_parameters()
        p_in, p_hidden = p
        if p_in < 0 or p_in > 1:
            raise ValueError("input dropout probability...")
        if p_hidden < 0 or p_hidden > 1:
            raise ValueError("hidden state dropout probability...")

        self.p_in = p_in
        self.p_hidden = p_hidden
        self.noise_in = None
        self.noise_hidden = None

    def reset_parameters(self):
        for weight in self.parameters():
            if weight.dim() == 1:
                nn.init.constant_(weight, 0.)
            else:
                self.initializer(weight.data)

    def reset_noise(self, batch_size):
        if self.training:
            if self.p_in:
                noise = self.weight_ih.new_empty(batch_size, self.input_size)
                self.noise_in = noise.bernoulli_(1.0 - self.p_in) / (1.0 - self.p_in)
            else:
                self.noise_in = None

            if self.p_hidden:
                noise = self.weight_hh.new_empty(batch_size, self.hidden_size)
                self.noise_hidden = noise.bernoulli_(1.0 - self.p_hidden) / (1.0 - self.p_hidden)
            else:
                self.noise_hidden = None
        else:
            self.noise_in = None
            self.noise_hidden = None

    def forward(self, input, hx):
        if self.nonlinearity == "tanh":
            func = rnn_F.VarRNNTanhCell
        elif self.nonlinearity == "relu":
            func = rnn_F.VarRNNReLUCell
        else:
            raise  RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity)
            )

        return func(
            input, hx, self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh, self.noise_in, self.noise_hidden,
        )


class VarMaskedRNN(VarMaskedRNNBase):
    def __init__(self, *args, **kwargs):
        super(VarMaskedRNN, self).__init__(VarRNNCell, *args, **kwargs)


class VarLSTMCell(VarRNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True, p=(0.5, 0.5), initializer=None):
        super(VarLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4, input_size, hidden_size))
        self.weight_hh = Parameter(torch.Tensor(4, hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4, hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4, hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.initializer = default_initializer(self.hidden_size) if initializer is None else initializer
        self.reset_parameters()
        p_in, p_hidden = p
        if p_in < 0 or p_in > 1:
            raise ValueError("input dropout probability has to be between 0 and 1, "
                             "but got {}".format(p_in))
        if p_hidden < 0 or p_hidden > 1:
            raise ValueError("hidden state dropout probability has to be between 0 and 1, "
                             "but got {}".format(p_hidden))
        self.p_in = p_in
        self.p_hidden = p_hidden
        self.noise_in = None
        self.noise_hidden = None

    def reset_parameters(self):
        for weight in self.parameters():
            if weight.dim() == 2:
                nn.init.constant_(weight, 0.)
            else:
                self.initializer(weight.data)

    def reset_noise(self, batch_size):
        if self.training:
            if self.p_in:
                noise = self.weight_ih.new_empty(4, batch_size, self.input_size)
                self.noise_in = noise.bernoulli_(1.0 - self.p_in) / (1.0 - self.p_in)
            else:
                self.noise_in = None

            if self.p_hidden:
                noise = self.weight_hh.new_empty(4, batch_size, self.hidden_size)
                self.noise_hidden = noise.bernoulli_(1.0 - self.p_hidden) / (1.0 - self.p_hidden)
            else:
                self.noise_hidden = None
        else:
            self.noise_in = None
            self.noise_hidden = None

    def forward(self, input, hx):
        return rnn_F.VarLSTMCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
            self.noise_in, self.noise_hidden,
        )

class VarMaskedLSTM(VarMaskedRNNBase):
    def __init__(self, *args, **kwargs):
        super(VarMaskedLSTM, self).__init__(VarLSTMCell, *args, **kwargs)
        self.lstm = True

class VarFastLSTMCell(VarRNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True, p=(0.5, 0.5), initializer=None):
        super(VarFastLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.initializer = default_initializer(self.hidden_size) if initializer is None else initializer
        self.reset_parameters()
        p_in, p_hidden = p
        if p_in < 0 or p_in > 1:
            raise ValueError("input dropout probability has to be between 0 and 1, "
                             "but got {}".format(p_in))
        if p_hidden < 0 or p_hidden > 1:
            raise ValueError("hidden state dropout probability has to be between 0 and 1, "
                             "but got {}".format(p_hidden))
        self.p_in = p_in
        self.p_hidden = p_hidden
        self.noise_in = None
        self.noise_hidden = None

    def reset_parameters(self):
        for weight in self.parameters():
            if weight.dim() == 1:
                nn.init.constant_(weight, 0.)
            else:
                self.initializer(weight.data)

    def reset_noise(self, batch_size):
        if self.training:
            if self.p_in:
                noise = self.weight_ih.new_empty(batch_size, self.input_size)
                self.noise_in = noise.bernoulli_(1.0 - self.p_in) / (1.0 - self.p_in)
            else:
                self.noise_in = None

            if self.p_hidden:
                noise = self.weight_hh.new_empty(batch_size, self.hidden_size)
                self.noise_hidden = noise.bernoulli_(1.0 - self.p_hidden) / (1.0 - self.p_hidden)
            else:
                self.noise_hidden = None
        else:
            self.noise_in = None
            self.noise_hidden = None

    def forward(self, input, hx):
        return rnn_F.VarFastLSTMCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
            self.noise_in, self.noise_hidden,
        )

class VarMaskedFastLSTM(VarMaskedRNNBase):
    def __init__(self, *args, **kwargs):
        super(VarMaskedFastLSTM, self).__init__(VarFastLSTMCell, *args, **kwargs)
        self.lstm = True

class VarGRUCell(VarRNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True, p=(0.5, 0.5), initializer=None):
        super(VarGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3, input_size, hidden_size))
        self.weight_hh = Parameter(torch.Tensor(3, hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3, hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3, hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.initializer = default_initializer(self.hidden_size) if initializer is None else initializer
        self.reset_parameters()
        p_in, p_hidden = p
        if p_in < 0 or p_in > 1:
            raise ValueError("input dropout probability has to be between 0 and 1, "
                             "but got {}".format(p_in))
        if p_hidden < 0 or p_hidden > 1:
            raise ValueError("hidden state dropout probability has to be between 0 and 1, "
                             "but got {}".format(p_hidden))
        self.p_in = p_in
        self.p_hidden = p_hidden
        self.noise_in = None
        self.noise_hidden = None

    def reset_parameters(self):
        for weight in self.parameters():
            if weight.dim() == 2:
                nn.init.constant_(weight, 0.)
            else:
                self.initializer(weight.data)

    def reset_noise(self, batch_size):
        if self.training:
            if self.p_in:
                noise = self.weight_ih.new_empty(3, batch_size, self.input_size)
                self.noise_in = noise.bernoulli_(1.0 - self.p_in) / (1.0 - self.p_in)
            else:
                self.noise_in = None

            if self.p_hidden:
                noise = self.weight_hh.new_empty(3, batch_size, self.hidden_size)
                self.noise_hidden = noise.bernoulli_(1.0 - self.p_hidden) / (1.0 - self.p_hidden)
            else:
                self.noise_hidden = None
        else:
            self.noise_in = None
            self.noise_hidden = None

    def forward(self, input, hx):
        return rnn_F.VarGRUCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
            self.noise_in, self.noise_hidden,
        )

class VarMaskedGRU(VarMaskedRNNBase):
    def __init__(self, *args, **kwargs):
        super(VarMaskedGRU, self).__init__(VarGRUCell, *args, **kwargs)
        self.lstm = True

class VarFastGRUCell(VarRNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True, p=(0.5, 0.5), initializer=None):
        super(VarFastGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.initializer = default_initializer(self.hidden_size) if initializer is None else initializer
        self.reset_parameters()
        p_in, p_hidden = p
        if p_in < 0 or p_in > 1:
            raise ValueError("input dropout probability has to be between 0 and 1, "
                             "but got {}".format(p_in))
        if p_hidden < 0 or p_hidden > 1:
            raise ValueError("hidden state dropout probability has to be between 0 and 1, "
                             "but got {}".format(p_hidden))
        self.p_in = p_in
        self.p_hidden = p_hidden
        self.noise_in = None
        self.noise_hidden = None

    def reset_parameters(self):
        for weight in self.parameters():
            if weight.dim() == 1:
                nn.init.constant_(weight, 0.)
            else:
                self.initializer(weight.data)

    def reset_noise(self, batch_size):
        if self.training:
            if self.p_in:
                noise = self.weight_ih.new_empty(batch_size, self.input_size)
                self.noise_in = noise.bernoulli_(1.0 - self.p_in) / (1.0 - self.p_in)
            else:
                self.noise_in = None

            if self.p_hidden:
                noise = self.weight_hh.new_empty(batch_size, self.hidden_size)
                self.noise_hidden = noise.bernoulli_(1.0 - self.p_hidden) / (1.0 - self.p_hidden)
            else:
                self.noise_hidden = None
        else:
            self.noise_in = None
            self.noise_hidden = None

    def forward(self, input, hx):
        return rnn_F.VarFastGRUCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
            self.noise_in, self.noise_hidden,
        )

class VarMaskedFastGRU(VarMaskedRNNBase):
    def __init__(self, *args, **kwargs):
        super(VarMaskedFastGRU, self).__init__(VarFastGRUCell, *args, **kwargs)