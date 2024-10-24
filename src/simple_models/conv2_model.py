import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2(nn.Module):
    def __init__(self, l2_reg):
        super(Conv2, self).__init__()

        self.conv2D_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2D_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc_1 = nn.Linear(64 * 16 * 16, 256)  
        self.fc_2 = nn.Linear(256, 256)
        self.fc_3 = nn.Linear(256, 10) 

        self.l2_reg = l2_reg

        nn.init.xavier_normal_(self.conv2D_1.weight)
        nn.init.xavier_normal_(self.conv2D_2.weight)
        nn.init.xavier_normal_(self.fc_1.weight)
        nn.init.xavier_normal_(self.fc_2.weight)
        nn.init.xavier_normal_(self.fc_3.weight)

    def forward(self, x):

        x = F.relu(self.conv2D_1(x))
        x = F.relu(self.conv2D_2(x))

        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))

        x = self.fc_3(x)

        return x