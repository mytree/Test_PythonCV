#0503.py

import cv2
import numpy as np

src = np.array([[0, 0, 0, 0],
                [1, 1, 3, 5],
                [6, 1, 1, 3],
                [4, 3, 1, 7]
                ], dtype = np.uint8)

# 4 x 4 배열 src 의 0번 채널에서 마스크 지정 없이,
#   히스토그램 빈 크기 histSize= [4], 범위 ranges = [0, 8] 로
#   히스토그램 hist1 계산. 범위[0,8] 에서 8은 포함하지 않는다.
hist1 = cv2.calcHist(images = [src], channels = [0], mask = None,
                     histSize = [4], ranges = [0, 8])
print('hist1 = ', hist1)

hist2 = cv2.calcHist(images = [src], channels = [0], mask = None,
                     histSize = [4], ranges = [0, 4])
print('hist2 = ', hist2)

