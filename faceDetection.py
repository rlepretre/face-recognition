# import numpy as np
import cv2



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cv2.namedWindow("Mac Camera")
vc = cv2.VideoCapture(0)

if vc.isOpened(): #pauses to make sure the frame can be read
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("Mac Camera", frame)
    rval, frame = vc.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC key
            break
    cv2.destroyWindow("Mac Camera")


