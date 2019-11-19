#0417.py

import cv2
import numpy as np

src1 = cv2.imread('../data/lena.jpg', cv2.IMREAD_GRAYSCALE)
src2 = np.zeros(shape = (512, 512), dtype = np.uint8) + 100

# numpy 의 배열 덧셈
# src1 + src2 결과가 255가 넘으면 256 나머지 계산
dst1 = src1 + src2

# 덧셈 결과가 255 넘으면 255로 계산
dst2 = cv2.add(src1, src2)

# dst2 의 화소 자료형을 dtype cv2.CV_8U 와 같이 명시 할 수 있다.
#dst2 = cv2.add(src1, src2, dtype = cv2.CV_8U)

cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()

