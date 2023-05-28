import cv2
path = '/Users/lanyiwei/data/ppt/images/10.jpg'
face = cv2.imread(path) # 读取单通道灰度图
print(face)
print(face.shape)
img_resized = cv2.resize(face, (128, 128), interpolation=cv2.INTER_CUBIC)
# # face_normalized = img_resized.reshape(1, 48, 48) / 255.0 # 为与pytorch中卷积神经网络API的设计相适配，需reshape原图
cv2.imshow('image', img_resized)
cv2.waitKey(10000)
cv2.destroyAllWindows()