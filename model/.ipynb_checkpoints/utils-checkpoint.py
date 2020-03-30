import torch
import copy
import json
from torch import nn
import numpy as np

def gen_attention_mask(token_ids, valid_length):
    attention_mask = torch.zeros_like(token_ids)
    for i, v in enumerate(valid_length):
        attention_mask[i][:v] = 1
    return attention_mask.float()

def generate_gen_dis_config(config):
    with open(config) as f:
        dis_param = json.load(f)
    
    gen_ratio = 1/dis_param['generator_ratio']
    gen_param = copy.deepcopy(dis_param)
    gen_param['hidden_size'] = int(dis_param['hidden_size'] * gen_ratio)
    gen_param['intermediate_size'] =int(dis_param['intermediate_size'] * gen_ratio)
    gen_param['num_attention_heads'] = int(dis_param['num_attention_heads'] * gen_ratio)
    return gen_param, dis_param

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
            )
        )
