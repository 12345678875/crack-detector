
import torch
import torch.nn as nn

class CrackCNN(nn.Module):
    def __init__(self , num_classes = 2):
        super(CrackCNN , self).__init__()
        self.conv1 = nn.Conv2d(3 , 16 , 3 ,padding = 1)
        self.pool = nn.MaxPool2d(2 , 2)
        self.conv2 = nn.Conv2d(16 , 32 , 3 , padding = 1)
        self.fc = nn.Linear(32 *56 * 56 , num_classes)
        self.relu = nn.ReLU()

    def forward(self , x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1 , 32 * 56 * 56)
        x = self.fc(self.relu(x))
        return x

#
#
def load_model(model_path , num_classes = 2 , device = 'cpu'):
    """加载训练好的模型"""
    model = CrackCNN(num_classes = num_classes)
    model.load_state_dict(torch.load(model_path , map_location = device))
    model.eval()
    return model
