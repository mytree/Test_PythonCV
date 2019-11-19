#0420.py
import cv2
import numpy as np

src = cv2.imread('../data/lena.jpg', cv2.IMREAD_GRAYSCALE)

# 최소, 최대, 최소 위치값, 최대 위치값
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(src)
print('src:', minVal, maxVal, minLoc, maxLoc)

# 최소/최대값 범위를 범위[100,200] 으로 정규화
dst = cv2.normalize(src, None, 100, 200, cv2.NORM_MINMAX)

# 최소, 최대, 최소 위치값, 최대 위치값 다시 계산
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dst)
print('dst:', minVal, maxVal, minLoc, maxLoc)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
