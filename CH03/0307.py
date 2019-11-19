#0307.py

import cv2
import numpy as np

img = np.zeros(shape=(512,512,3),dtype=np.uint8) + 255

x, y = 256, 256
size = 200

for angle in range(0, 90, 10):
    rect = ((256, 256), (size, size), angle)

    # cv2.boxPoints()로 중심(256,256), 크기(size,size), 각도(angle)인
    #   회전 사각형 rect의 모서리 점을 정수로 변환하여 box에 계산한다.
    box = cv2.boxPoints(rect).astype(np.int32)
    r = np.random.randint(256)
    g = np.random.randint(256)
    b = np.random.randint(256)

    # cv2.polylines()함수로 회전 사각형의 모서리 점을 닫힌 사각형을 그린다.
    #   난수로 생성한 r,g,b 를 색상으로 하여 사각형을 그린다.
    cv2.polylines(img, [box], True, (r,g,b), 2)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()



        
