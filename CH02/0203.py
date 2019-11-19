# 0203.py
import cv2
from matplotlib import pyplot as plt


imageFile = '../data/lena.png'
imgBGR = cv2.imread(imageFile)   # cv2.IMREAD_COLOR
plt.axis('off')     # x, y축을 표시하지 않는다.
# plt.imshow(imgBGR)
# plt.show()


# 채널 순서를 RGB 로 변경
#   Matplotlib 는 RGB 채널순서로 처리함
imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

plt.imshow(imgRGB)
plt.show()
