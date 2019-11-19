#0202.py
import cv2

imageFile = '../data/lena.png'
img = cv2.imread(imageFile) # cv2.imread(imageFile, cv2.IMREAD_COLOR)
cv2.imwrite('./data/Lena.bmp', img)
cv2.imwrite('./data/Lena.jpg', img)

# [cv2.IMWRITE_PNG_COMPRESSION, 9] 는
#   img 를 압축률 9의 PNG 이미지로 'Lena2.png' 파일에 저장한다.
#   압축률의 범위는 [0, 9]이며 압축률이 높을 수록 시간이 오래 걸린다.
#   기본값은 3이다.
cv2.imwrite('./data/Lena2.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

# img 를 90% 의 품질을 갖는 JPEG 이미지로 저장
#   품질의 범위는 [0, 100] 이며 높을 수록 품질이 좋다.
#   기본값은 95이다.
cv2.imwrite('./data/Lena2.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
