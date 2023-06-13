import requests
import numpy as np
import cv2
from hiabao_classifier import classifier
# 拿图片
# response = requests.get('https://5b0988e595225.cdn.sohucs.com/images/20180926/263028cf94ab487abb94c2ccfc61ede1.jpeg')
response = requests.get('https://scpic.chinaz.net/files/pic/pic9/201212/xpic8895.jpg')
# response = requests.get('https://img2.baidu.com/it/u=1009535824,2117587501&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=712')
path = '/Users/lanyiwei/data/other/12.jpg'
path = r'Z:\data\image_dp\other\1.jpg'
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


