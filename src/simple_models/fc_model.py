import torch
import torch.nn as nn
import torch.nn.functional as F

class NetSimple(nn.Module):
    def __init__(self):
        super(NetSimple, self).__init__()
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 
    
    def load_weights(self, path, map_location=None):
        weights = torch.load(path, map_location=map_location)

        self.fc1.weight.data.copy_(weights['fc1_weight'])
        self.fc2.weight.data.copy_(weights['fc2_weight'])
        self.fc3.weight.data.copy_(weights['fc3_weight'])

        if 'fc1_bias' in weights:
            self.fc1.bias.data.copy_(weights['fc1_bias'])
        if 'fc2_bias' in weights:
            self.fc2.bias.data.copy_(weights['fc2_bias'])
        if 'fc3_bias' in weights:
            self.fc3.bias.data.copy_(weights['fc3_bias'])

        print(f"Weights loaded from {path}")
    