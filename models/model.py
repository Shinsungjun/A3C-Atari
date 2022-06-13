import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#A3C Neural Network Model
class A3C(nn.Module):

    def __init__(self, num_inputs, action_space):
        super(A3C, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.gru = nn.GRUCell(800, 256) 

        num_outputs = 3
        '''
        action 1 -> stay
        action 2 -> up
        action 3 -> down
        '''
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        # self.gru.bias_ih.data.fill_(0)
        # self.gru.bias_hh.data.fill_(0)

        self.train()

    def forward(self, state_, hx_):
        inputs, hx = state_, hx_
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x)) 
        x = F.relu(self.conv5(x)) 
        # print(x.shape)
        #x = F.relu(self.conv5(x)) # 1 x 32 x 5 x 5
        x = x.view(-1, 800)
        hx = self.gru(x, hx)
        x = hx
        
        critic_output = self.critic_linear(x)
        action_output = self.actor_linear(x)
        
        return critic_output, action_output, hx
