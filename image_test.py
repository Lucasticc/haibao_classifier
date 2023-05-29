import requests
import numpy as np
import cv2
from hiabao_classifier import classifier
# 拿图片
response = requests.get('https://cdn.pixabay.com/photo/2023/05/09/17/20/flowers-7982037_1280.jpg')
response = requests.get('https://preview.qiantucdn.com/auto_machine/20230327/32b1ef5d-5851-4e10-912e-20f3626aa6b9.jpg!w1024_new_small_1')
path = '/Users/lanyiwei/data/image/11.jpg'
image_data = response.content

# 二进制赚np
nparr = np.frombuffer(image_data, np.uint8)
img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
img_cv = cv2.imread(path)
# print(img_cv)
# print('/n')
# print(img_np)
# if img_cv == img_cv:
# # 显示图像
#     cv2.imshow('image', img_np)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print('no')
clf = classifier()
result = clf.upload(img_np)
print(result)


