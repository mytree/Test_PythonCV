#0308.py

import cv2
import numpy as np

img = np.zeros(shape=(512,512,3),dtype=np.uint8)+255

pts1 = np.array([[100,100],[200,100],[200,200],[100,200]])
pts2 = np.array([[300,200],[400,100],[400,200]])

# cv2.fillConvexPoly()로 하나의 볼록 다각형 pts1을 (255,0,0)색상으로 채운다.
cv2.fillConvexPoly(img, pts1, color=(255,0,0))

# cv2.fillPoly로 다각형 pts2의 배열[pts2]를 (0,0,255) 색상으로 채운다.
#   pts1 과 pts2 를 리스트 []로 감싸지 않으면 오류가 발생한다.
#   pts1 과 pts2 의 배열 [pts1,pts2]를 사용하면 한 번에 2개의 다각형을 채운다.
cv2.fillPoly(img,[pts2], color=(0,0,255))
#cv2.fillPoly(img,[pts1, pts2], color=(0,0,255))

cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()

