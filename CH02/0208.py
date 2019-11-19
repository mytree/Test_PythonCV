#0208.py
import cv2

## cap = cv2.VideoCapture('http://172.30.1.18:4747/mjpegfeed')     # droid cam
## cap = cv2.VideoCapture('http://172.30.1.18.4747/mjpegfeed?640x480')
## cap = cv2.VideoCapture('http://172.30.1.18.8080/video')         # ip WebCam

cap = cv2.VideoCapture('rtsp://192.168.1.162:1935/vod/bmt_side.mp4')

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('frame_size =', frame_size)

while True:
    retval, frame = cap.read()      # 프레임 캡처
    if not retval:
        break

    cv2.imshow('frame', frame)

    key = cv2.waitKey(25)
    if key == 27:           # ESC key
        break
    
if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()

    
