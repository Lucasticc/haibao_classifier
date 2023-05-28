import torch
import cv2
import torch.nn as nn

class classifier():
    def __init__(self) -> None:
        self.model = BinaryClassifier()
        self.model.load_state_dict(torch.load('/Users/lanyiwei/data/ppt/model.pth'))
        self.model.eval

    def upload(self,pic):
        face = cv2.resize(pic, (128, 128), interpolation=cv2.INTER_CUBIC)
        face_normalized = face.reshape(3, 128, 128) / 255.0 # 为与pytorch中卷积神经网络API的设计相适配，需reshape原图
        face_tensor = torch.from_numpy(face_normalized) # 将python中的numpy数据类型转化为pytorch中的tensor数据类型
        face_tensor = face_tensor.type('torch.Tensor') # 指定为'torch.FloatTensor'型，否则送进模型后会因数据类型不匹配而报错
        face_tensor = face_tensor.unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(face_tensor)
        print(outputs)
        if outputs.item()<0.5:
            return('是')
        else:
            return('不是')
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(-1, 64 * 32 * 32)
        x = self.relu3(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
path = '/Users/lanyiwei/data/image/97.jpeg'
face = cv2.imread(path)
# model = BinaryClassifier()
# model.load_state_dict(torch.load('/Users/lanyiwei/data/ppt/model.pth'))
# model.eval

clas = classifier()
clas(face)