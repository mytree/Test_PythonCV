#0205.py
import cv2
from matplotlib import pyplot as plt

imageFile = '../data/lena.jpg'
imgGray = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)

# 크기를 (6인치, 6인치)로 설정
plt.figure(figsize=(6,6))

# 영상 출력 범위를 좌우[0,1], 위아래[0,1]로 조정
# 범위는 left<right 와 bottom<top 이어야 한다.
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

# imgGray 영상을 그레이스케일로 화면에 표시(cmap='gray')
plt.imshow(imgGray, cmap='gray')

##plt.axis('tight')
plt.axis('off')

# 영상을 '0205.png'파일에 저장
plt.savefig('./data/0205.png')

plt.show()

