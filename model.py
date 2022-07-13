import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN(nn.Module):

    def __init__(self, image_shape, channels, actions):
        super(CNN, self).__init__()
        self.num_actions = actions

        self.conv1 = nn.Conv2d(channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        c_out = self.conv3(self.conv2(self.conv1(torch.randn(1, channels, * image_shape))))
        self.conv3_size = np.prod(c_out.shape)

        self.fc1 = nn.Linear(self.conv3_size, 512)
        self.fc2 = nn.Linear(512, actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, self.conv3_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
