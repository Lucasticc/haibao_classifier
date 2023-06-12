import torch
import torch.nn as nn
import torch.optim as optim
from model import BinaryClassifier
from dataloader import FaceDataset
from net import Net
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
train_dataset = '/Users/lanyiwei/data/ppt'
# train_dataset = r'Z:\data\image_dp'
train_dataset = FaceDataset(train_dataset)
test_dataset = '/Users/lanyiwei/data/ppt'
# test_dataset = r'Z:\data\val_dp'

# 定义数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

# 定义模型、损失函数和优化器
model = BinaryClassifier()
# model = Net()
path_win = r'Z:\data\ppt_model\model50-3.pth'
# model.load_state_dict(torch.load(path_win))
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
model.train()
model.to(device)
for epoch in range(1,20):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # inputs = inputs.view(-1, 3*48*48)

        optimizer.zero_grad()
        inputs=inputs.to(device)
        outputs = model(inputs)
        labels = labels.unsqueeze(1)

        labels = labels.float()
        labels = labels.to(device)
        criterion.to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    large = len(train_loader)
    lossp = running_loss/large
    print('第%d轮,损失函数%5f'%(epoch,lossp))

        # if i%(large-1) == 0 and i !=0:
        #     lossp = running_loss/large
        #     print('第%d轮,损失函数%5f'%(epoch,lossp))
        #     running_loss = 0.0

test_dataset = FaceDataset(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
total = len(test_dataset)
with torch.no_grad():
    right = 0 
    for data in test_loader:
        images, labels = data
        model.eval()
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        threshold = 0.5
        predicted = torch.where(outputs >= threshold, 1, 0)
        right = (predicted == labels).sum().item()
        # right = right + (outputs.argmax(1)==labels).sum()  # 计数
    acc = right/total
print(acc)
print('Accuracy of the network on the test images: %d %%' % (acc))
path = '/Users/lanyiwei/data/ppt/model.pth'
path = r'Z:\data\ppt_model\model5-4.pth'
torch.save(model.state_dict(), path)