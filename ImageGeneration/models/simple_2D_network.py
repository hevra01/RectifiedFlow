from itertools import tee
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils, layers

# Custom implementation of pairwise for versions before Python 3.10
def pairwise(iterable):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

@utils.register_model(name='Simple2DNetwork')
class Simple2DNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()

        input_dim = config.model.input_dim
        hidden_dims = config.model.hidden_dims
        output_dim = config.model.output_dim

        layers = []
        for in_dim, out_dim in pairwise((input_dim,) + hidden_dims):
            
            # here, we use extend because we are concatenating 2
            # lists. we have a list because we have the linear layer
            # and activation together.
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
        
        # we don't append the activation here.
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # self.net is defined as a sequential container of the constructed layers.
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, time_cond):
        temb = layers.get_timestep_embedding(time_cond)
        x = torch.cat([x, temb], dim=1)

        output = self.net(x)
        return output

