import torch
from torch import nn

class ChatBot(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatBot, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size),
        )

    def forward(self,x):
        return self.seq(x)