#0419.py
import cv2
import numpy as np

src1 = cv2.imread('../data/lena.jpg', cv2.IMREAD_GRAYSCALE)
src2 = np.zeros(shape = (512, 512), dtype = np.uint8) + 255

# numpy 의 브로드캐스팅, 255를 src1 크기의 배열로 확장하고,
#  src1 의 각 화소와 뺄셈을 계산하여 src1 영상을 반전영상으로 만듦
dst1 = 255 - src1

dst2 = cv2.subtract(src2, src1)

# dst1 과 dst2 의 각 화소를 비교, 참이면 255, 거짓이면 0 을
#   dst 영상 각화소에 출력
dst3 = cv2.compare(dst1, dst2, cv2.CMP_NE)      # cv2.CMP_EQ

#   0 이 아닌 화소를 카운트하여 반환한다.
n = cv2.countNonZero(dst3)

print('n=',n)

cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()

