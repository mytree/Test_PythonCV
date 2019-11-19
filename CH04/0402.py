#0402.py

import cv2
# import numpy as np

img = cv2.imread('../data/lena.jpg', cv2.IMREAD_GRAYSCALE)
print('img.shape=', img.shape)

# img.reshape(-1, 512, 512) 는 3차원 배열로 확장 -1 은 자동 계산
#   실제 데이터는 변경하지 않고 모양을 변경한다.
# img = img.reshape(img.shape[0]*img.shape[1])

# img.flatten() 은 다차원 배열을 1차원 배열로 변경하는 함수
img = img.flatten()
print('img.shape=', img.shape)

img = img.reshape(-1, 512, 512)
print('img.shape=', img.shape)

cv2.imshow('img', img[0])   # 그레이 스케일 영상을 화면에 표시
cv2.waitKey()
cv2.destroyAllWindows()
