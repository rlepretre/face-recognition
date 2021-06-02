import cv2
import numpy as np
import os
from helpers import normalize

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

for filename in os.listdir('myFaceDB/originals'):
    if filename.endswith(".jpg"):
        print('processing: ' + filename)
        img = cv2.imread('myFaceDB/originals/' + filename)
        img, left_eye, theta, eyes_dist = normalize(img)
        if img is not None:
            cv2.imwrite('myFaceDB/normalized/' + filename, img)


