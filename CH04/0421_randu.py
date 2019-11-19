#0421.py
import cv2
import numpy as np
import time

dst = np.full((512, 512, 3), (255, 255, 255), dtype = np.uint8)
nPoints = 100
pts = np.zeros((1, nPoints, 2), dtype = np.uint16)

# 난수 생성 초기화, 초기화 안하면 항상 같은 난수열을 생성함
cv2.setRNGSeed(int(time.time()))

# 1xnPoints & 2채널 인 pts 배열에 (0,0) ~ (512, 512) 범위의 난수를 생성
cv2.randu(pts, (0,0), (512, 512))

# draw points
for k in range(nPoints):
    # pts[0, k]의 채널 데이터를 x, y에 저장
    x, y = pts[0, k][:]         # pts[0, k, :]
    cv2.circle(dst, (x, y), radius = 5, color = (0, 0, 255), thickness = -1)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()

