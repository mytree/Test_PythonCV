#0309.py

import cv2
import numpy as np

img = np.zeros(shape=(512,512,3),dtype=np.uint8)+ 255
text = 'OpenCV Programming'
org = (50, 100)
font = cv2.FONT_HERSHEY_SIMPLEX

# cv2.putText()로 text 문자열을 org 위치에 font 폰트, 폰트 스케일 1,
#   (255,0,0)색상, 두께2로 출력한다.
cv2.putText(img, text, org, font, 1, (255, 0, 0), 2)

# cv2.getTextSize()로 text문자열을 font 폰트, 폰트 스케일1, 두께 2로
#   출력하기 위한 사각형의 크기를 size에 반환하고,
#   baseLine은 사각형 아래 기준선의 상대적 y값을 반환한다.
#   print()문은 size=(345,22), baseLine=10을 출력한다.
size, baseLine = cv2.getTextSize(text, font, 1, 2)
print('org=',org)               # org= (50,100)
print('size=',size)             # size= (345,22)
print('baseLine=',baseLine)     # baseLine= 10

# cv2.rectangle()은 사각형 모서리 좌표 org, (org[0]+size[0], org[1]-size[1])로
#   문자열의 출력 위치를 그린다. 실제 기준선의 y좌표는 org[1] + baseLine 이다.
cv2.rectangle(img, org, (org[0] + size[0], org[1] - size[1]), (0, 0, 255))
#cv2.rectangle(img, (org[0],org[1]-size[1]),(org[0]+size[0],org[1]+baseLine), (0,0,255))
cv2.circle(img, org, 3, (0,255,0), 2)

cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()
