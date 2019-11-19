
#0404.py

import cv2
## import numpy as np

img = cv2.imread('../data/lena.jpg')     # cv2.IMREAD_COLOR

# y = 100행, x=200열의 영상 화소의 컬러를
#   리스트[255,0,0] 또는 튜플(255,0,0)으로 변경
#   컬러는 [B,G,R] 채널 순서
img[100, 200] = [255, 0, 0]             # 컬러(BGR) 변경

# numpy 슬라이싱, y=100, x=200 에서 x=209까지의
#   1x10 영역을 ROI 로 지정하여 컬러를 출력
#print(img[100, 200:210])                # ROI 접근

# y=100 에서 y=399까지, x=200에서 x=299까지 사각 영역의 화소를
#   파란색으로 변경
#for y in range(100,400):
#    for x in range(200,300):
#        img[y,x] = [255, 0, 0]          # 파란색(blue)으로 변경

img[100:400, 200:300] = [255,0,0]       # ROI 접근

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()

