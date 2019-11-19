#0210.py

import cv2

# cap = cv2.VideoCapture(0)   # 0번 카메라
cap = cv2.VideoCapture('rtsp://192.168.1.162:1935/vod/bmt_side.mp4')

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('frame_size = ', frame_size)

# VideoWriter_fourcc(*'XVID')로 코덱을 fourcc 에 생성
# fourcc = cv2.VideoWriter_fourcc(*'DIVX') # ('D','I','V','X')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')

fourcc = cv2.VideoWriter_fourcc(*'H264')

out1 = cv2.VideoWriter('../data/record0.mp4', fourcc, 20.0, frame_size)
out2 = cv2.VideoWriter('../data/record1.mp4', fourcc, 20.0, frame_size, isColor = False)

while True:
    retval, frame = cap.read()
    if not retval:
        break
    out1.write(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out2.write(gray)
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)

    key = cv2.waitKey(25)
    if key == 27:
        break

cap.release()
out1.release()
out2.release()
cv2.destroyAllWindows()


    
