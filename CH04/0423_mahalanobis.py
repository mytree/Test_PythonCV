# 0423.py
import cv2
import numpy as np

X = np.array([[0, 0, 0, 100, 100, 150, -100, -150],
              [0, 50, -50, 0, 30, 100, -20, -100]], dtype = np.float64)

# 전치행렬 변경
X = X.transpose()       # X = X.T

cov, mean = cv2.calcCovarMatrix(X, mean = None,
                                flags = cv2.COVAR_NORMAL + cv2.COVAR_ROWS)
print('mean=', mean)
print('cov=', cov)

# 공분산 행렬 cov 의 역행렬 icov 계산
ret, icov = cv2.invert(cov)
print('icov=', icov)

v1 = np.array([[0],[0]], dtype = np.float64)
v2 = np.array([[0],[50]], dtype = np.float64)

# 벡터 v1, v2 사이의 마하라노비스 통계적 거리는 공분산 행렬의
#   역행렬을 이용해서 계산
dist = cv2.Mahalanobis(v1, v2, icov)
print('dist = ', dist)

cv2.waitKey()
cv2.destroyAllWindows()
