import torch
import torch.nn as nn
import torch.optim as optim
from model import BinaryClassifier
from dataloader import FaceDataset

# 加载数据集
train_dataset = '/Users/lanyiwei/data/ppt'
train_dataset = FaceDataset(train_dataset)
test_dataset = '/Users/lanyiwei/data/ppt'
test_dataset = FaceDataset(test_dataset)

# 定义数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# 定义模型、损失函数和优化器
model = BinaryClassifier()
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
model.train()
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # inputs = inputs.view(-1, 3*48*48)

        optimizer.zero_grad()

        outputs = model(inputs)
        labels = labels.unsqueeze(1)

        labels = labels.float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

total = len(test_loader)
with torch.no_grad():
    right = 0 
    for data in test_loader:
        images, labels = data
        model.eval()
        outputs = model(images)
        right = right + (outputs.argmax(1)==labels).sum()  # 计数
        acc = right.item()/total

print('Accuracy of the network on the test images: %d %%' % (acc))
torch.save(model.state_dict(), '/Users/lanyiwei/data/ppt/model.pth')