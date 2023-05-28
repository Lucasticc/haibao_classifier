import torch
from model import BinaryClassifier
import cv2
model = BinaryClassifier()
model.load_state_dict(torch.load('/Users/lanyiwei/data/ppt/model.pth'))
model.eval()
path = '/Users/lanyiwei/data/image/97.jpeg'
face = cv2.imread(path)
face = cv2.resize(face, (128, 128), interpolation=cv2.INTER_CUBIC)
face_normalized = face.reshape(3, 128, 128) / 255.0 # 为与pytorch中卷积神经网络API的设计相适配，需reshape原图
face_tensor = torch.from_numpy(face_normalized) # 将python中的numpy数据类型转化为pytorch中的tensor数据类型
face_tensor = face_tensor.type('torch.Tensor') # 指定为'torch.FloatTensor'型，否则送进模型后会因数据类型不匹配而报错
face_tensor = face_tensor.unsqueeze(0)
with torch.no_grad():
    outputs = model(face_tensor)
print(outputs)
if outputs.item()<0.5:
    print('是')
else:
    print('滚')