import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from dataset import CrackDataset
from models.crack_cnn import CrackCNN

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_path = r"C:\Users\16222\OneDrive\桌面\AI_Project\CrackForest-dataset"

image_dir = os.path.join(data_path,'image')
gt_dir = os.path.join(data_path,'groundTruth')

dataset = CrackDataset(image_dir,gt_dir,transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset , test_dataset = torch.utils.data.random_split(
    dataset,[train_size,test_size]
)

train_loader = DataLoader(train_dataset,batch_size=8,shuffle=True)

model = CrackCNN()

criterion = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):

    model.train()

    running_loss = 0

    for images , labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{num_epochs} Loss:{avg_loss:.4f}")

print("训练完成")