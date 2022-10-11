import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

class CNN_RNN(nn.Module):

    def __init__ (self):
        super(CNN_RNN, self).__init__()

        self.cnn = models.mobilenet_v2(pretrained=True)
        self.cnn.classifier[1] = nn.Linear(1280, 512)
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.linear = nn.Linear(128 * 2, 3)

    def forward(self, x):
        x2 = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
        x2 = self.cnn(x2)
        r_in = x2.view(x.size(0), x.size(1), -1)
        h_0 = torch.zeros(self.rnn.num_layers * 2, x.size(0), self.rnn.hidden_size).to(x.device)
        c_0 = torch.zeros(self.rnn.num_layers * 2, x.size(0), self.rnn.hidden_size).to(x.device)
        r_out, _ = self.rnn(r_in, (h_0, c_0))
        out = self.linear(r_out[:, -1, :])
        return out

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if state_dict[name].size() != own_state[name].size():
                    print('Skip loading parameter {}'.format(name))
                    continue
                own_state[name].copy_(param)


class CNN(nn.Module):

    def __init__ (self):
        super(CNN, self).__init__()

        self.cnn = models.mobilenet_v2(pretrained=True)
        self.cnn.classifier[1] = nn.Linear(1280, 512)
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.out(x)
        return x


"""
import pdb
x = torch.randn(4, 11, 3, 224, 224).cuda()
model = CNN_RNN().cuda()
out = model(x)
pdb.set_trace()
"""
