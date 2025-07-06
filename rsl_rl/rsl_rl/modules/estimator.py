from turtle import forward
import numpy as np
from rsl_rl.modules.actor_critic import get_activation

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU
from torch.nn.utils.parametrizations import spectral_norm

class Estimator(nn.Module):
    def __init__(self,  input_dim,
                        output_dim,
                        hidden_dims=[256, 128, 64],
                        activation="elu",
                        **kwargs):
        super(Estimator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        activation = get_activation(activation)
        estimator_layers = []
        estimator_layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        estimator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                estimator_layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                estimator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                estimator_layers.append(activation)
        # estimator_layers.append(nn.Tanh())
        self.estimator = nn.Sequential(*estimator_layers)
    
    def forward(self, input):
        return self.estimator(input)
    
    def inference(self, input):
        with torch.no_grad():
            return self.estimator(input)

# 分类器，用于区分不同的技能或状态。它通常用于强化学习中的技能发现或对抗学习
class Discriminator(nn.Module):
    def __init__(self, n_states, 
                 n_skills, 
                 hidden_dims=[256, 128, 64], 
                 activation="elu"):
        super(Discriminator, self).__init__()
        self.n_states = n_states    # 输入状态的维度
        self.n_skills = n_skills    # 输出技能的维度（分类数量）

        activation = get_activation(activation)
        discriminator_layers = []
        discriminator_layers.append(nn.Linear(n_states, hidden_dims[0]))
        discriminator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                discriminator_layers.append(nn.Linear(hidden_dims[l], n_skills))
            else:
                discriminator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                discriminator_layers.append(activation)
        self.discriminator = nn.Sequential(*discriminator_layers)
        # self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        # init_weight(self.hidden1)
        # self.hidden1.bias.data.zero_()
        # self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        # init_weight(self.hidden2)
        # self.hidden2.bias.data.zero_()
        # self.q = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
        # init_weight(self.q, initializer="xavier uniform")
        # self.q.bias.data.zero_()

    def forward(self, states):
        return self.discriminator(states)

    def inference(self, states):
        with torch.no_grad():
            return self.discriminator(states)

# 一种特殊的分类器，使用了 spectral_norm（谱归一化）来稳定训练过程，通常用于对抗学习。
class DiscriminatorLSD(nn.Module):
    def __init__(self, n_states, 
                 n_skills, 
                 hidden_dims=[256, 128, 64], 
                 activation="elu"):
        super(DiscriminatorLSD, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills

        activation = get_activation(activation)
        discriminator_layers = []
        discriminator_layers.append(spectral_norm(nn.Linear(n_states, hidden_dims[0])))
        discriminator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                discriminator_layers.append(spectral_norm(nn.Linear(hidden_dims[l], n_skills)))
            else:
                discriminator_layers.append(spectral_norm(nn.Linear(hidden_dims[l], hidden_dims[l + 1])))
                discriminator_layers.append(activation)
        self.discriminator = nn.Sequential(*discriminator_layers)
        

    def forward(self, states):
        return self.discriminator(states)

    def inference(self, states):
        with torch.no_grad():
            return self.discriminator(states)

# 一种连续技能分类器，通常用于 DIAYN（Diversity Is All You Need）算法。它的输出经过归一化处理。
class DiscriminatorContDIAYN(nn.Module):
    def __init__(self, n_states, 
                 n_skills, 
                 hidden_dims=[256, 128, 64], 
                 activation="elu"):
        super(DiscriminatorContDIAYN, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills

        activation = get_activation(activation)
        discriminator_layers = []
        discriminator_layers.append(nn.Linear(n_states, hidden_dims[0]))
        discriminator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                discriminator_layers.append(nn.Linear(hidden_dims[l], n_skills))
            else:
                discriminator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                discriminator_layers.append(activation)
        self.discriminator = nn.Sequential(*discriminator_layers)

    def forward(self, states):
        return torch.nn.functional.normalize(self.discriminator(states))

    def inference(self, states):
        with torch.no_grad():
            return self.discriminator(states)