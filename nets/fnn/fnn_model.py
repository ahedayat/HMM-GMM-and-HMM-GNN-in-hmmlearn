import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, layers_neuron_num, soft_max=False):
        super(FNN, self).__init__()
        modules = []
        for (prev_neuron_num, crr_neuron_num) in zip(layers_neuron_num[:-1],layers_neuron_num[1:]) :
            modules.append( nn.Linear(prev_neuron_num, crr_neuron_num) )
        self.net = nn.Sequential( *modules )
        self.use_softmax = soft_max
        self.soft_max = nn.Sequential()
        if soft_max:
            self.soft_max = nn.Softmax()

    def forward(self, x):
        x = self.net(x)
        x = self.soft_max(x)
        return x
