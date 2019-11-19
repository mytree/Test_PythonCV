#0204.py

import cv2
from matplotlib import pyplot as plt

imageFile = '../data/lena.jpg'

# 그레이스케일로 영상 읽어서 imgGray 에 할당
# 컬러영상으로 읽어 cvtColor() 함수를 이용해서 그레이스케일 영상으로 변경할 수도 있다.
imgGray = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)

# x,y 축을 표시하지 않으려면 'off', 아니면 'on'
plt.axis('off')

# imgGray 영상을 'gray' 컬러맵(cmap), 'bicubic'으로 보간하여 표시
plt.imshow(imgGray,cmap = "gray", interpolation='bicubic')
plt.show()
