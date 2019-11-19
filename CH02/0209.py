#0209.py

'''
pip install youtube_dl
pip install pafy
'''

import cv2, pafy

# Q7Dk 다음은 소문자 엘, 7A 다음은 대문자 아이
url = 'https://www.youtube.com/watch?v=u_Q7Dkl7AIk' 

# 유투브 url 로부터 pafy.backend_youtube_dl.YtdlPafy 클래스 객체 video 생성
# 비디오 타이틀, 등급, 재생시간 등 출력
video = pafy.new(url)
print('title=',video.title)
print('video.rating=',video.rating)
print('video.duration=',video.duration)

# 비디오 양식파일 'webm' 최적의 pafy.beckend_youtube_dl.YtdlStream
#   클래스 객체 best 를 생성, 비디오 해상도 출력
best = video.getbest(preftype = 'webm')     # 'mp4', '3gp'
print('best.resolution', best.resolution)

# best.url 을 사용하여 VideoCapture 객체 cap 생성
cap = cv2.VideoCapture(best.url)
while(True):
    retval, frame = cap.read()
    if not retval:
        break
    cv2.imshow('frame',frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)   # 에지 검출
    cv2.imshow('edges', edges)

    key = cv2.waitKey(25)
    if key == 27:       # ESC key
        break
cv2.destroyAllWindows()
