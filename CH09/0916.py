# 0916.py
import cv2
import numpy as np

#1
src1 = cv2.imread('../data/book1.jpg')
src2 = cv2.imread('../data/book2.jpg')
img1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

#2
surF = cv2.xfeatures2d.SURF_create()
kp1, des1 = surF.detectAndCompute(img1, None)
kp2, des2 = surF.detectAndCompute(img2, None)
print('len(kp1)={}, len(kp2)={}'.format(len(kp1), len(kp2)))

#3
distT = 0.1
flan = cv2.FlannBasedMatcher_create()
matches = flan.radiusMatch(des1, des2, maxDistance=distT)
print('len(matches)=', len(matches))

#4
good_matches = []
for i, radius_match in enumerate(matches):
    if len(radius_match) != 0:
        for m in radius_match:
            good_matches.append(m)

print('len(good_matches)=', len(good_matches))

#5
src1_ptr = np.float32([kp1[m.queryIdx].pt for m in good_matches])
src2_ptr = np.float32([kp2[m.trainIdx].pt for m in good_matches])

H, mask = cv2.findHomography(src1_ptr, src2_ptr, cv2.RANSAC, 3.0)   # cv2.LMEDS
mask_matches = mask.ravel().tolist()        # list(mask.flattern())

#6
h, w = img1.shape
pts = np.float32([ [0, 0], [0, h-1], [w-1, h-1], [w-1,0] ]).reshape(-1, 1, 2)
pts2 = cv2.perspectiveTransform(pts, H)
src2 = cv2.polylines(src2, [np.int32(pts2)], True, (255, 0, 0), 2)

draw_params = dict(matchColor = (0, 255, 0), singlePointColor=None, matchesMask=mask_matches, flags= 2)
dst3 = cv2.drawMatches(src1, kp1, src2, kp2, good_matches, None, **draw_params)
cv2.imshow('dst3', dst3)
cv2.waitKey()
cv2.destroyAllWindows()



