import torch.nn as nn

class CrackCNN(nn.Module):

    def __init__(self):

        super(CrackCNN, self).__init__()

        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.pool2 = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(32*56*56,128)
        self.fc2 = nn.Linear(128,2)

        self.relu = nn.ReLU()

    def forward(self,x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = x.view(-1,32*56*56)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x