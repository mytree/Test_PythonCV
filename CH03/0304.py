#0304.py
import cv2
import numpy as np

img = np.zeros(shape=(512,512,3), dtype=np.uint8) + 255
ptCenter = img.shape[0] // 2, img.shape[1] // 2
size = 200, 100

# cv2.ellipse() 로 타원의 중심, 축 절반 크기, 각도 0, 시작 각도 0, 끝각도 360
#   색상 (0, 0, 255) 인 타원을 그린다.
cv2.ellipse(img, ptCenter, size, 0, 0, 360, (255, 0, 0))

# cv2.ellipse() 로 타원의 중심, 축 절반 크기, 각도 45, 색상(0, 0, 255) 타원 그림
cv2.ellipse(img, ptCenter, size, 45, 0, 360, (0, 0, 255))

# 중심 ptCenter, 크기 size, 각도 0 인 box 로 색상(255, 0, 0), 두께 5 타원 그림
box = (ptCenter, size, 0)
cv2.ellipse(img, box, (255, 0, 0), 5)

# 중심 ptCenter, 크기 size, 각도 45 인 box 로 색상 (0, 0, 255), 두께5인 타원 그림
box = (ptCenter, size, 45)
cv2.ellipse(img, box, (0, 0, 255), 5)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()

