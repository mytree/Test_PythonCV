#0412.py
import cv2
src = cv2.imread('../data/lena.jpg')

# 3 채널 컬러영상 src 를 채널 분리하여 b, g, r 에 저장한다.
b,g,r = cv2.split(src)

# 리스트 [b, g, r] 을 dst 에 채널 합성한다.
# 리스트의 항목의 순서 b, g, r의 순서는 채널 순서로 중요하다.
# cv2.merge([r,g,b])는 다른 색상의 컬러영상을 생성한다.
# type(dst) 는 numpy.ndarray 이다.
# dst.shape = (512, 512, 3) 로 dst 는 src 와 같은 3 채널 컬러영상이다.
dst = cv2.merge([b,g,r])    # cv2.merge([r,g,b])

print(type(dst))
print(dst.shape)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()

