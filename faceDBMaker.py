import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

for filename in os.listdir('myFaceDB/originals'):
    if filename.endswith(".jpeg"):
        print('processing: ' + filename)
        img = cv2.imread('myFaceDB/originals/' + filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.4, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3)
            eyes_centers = []

            for (ex,ey,ew,eh) in eyes:
                eyes_centers.append(np.array((x+(ex + ew//2),y + (ey + eh//2))))
                
            
            #Finds the angle theta between both eyes
            hyp = np.linalg.norm(eyes_centers[0]-eyes_centers[1])
            theta = -np.arcsin((eyes_centers[0][1]-eyes_centers[1][1])/hyp)
            scale = 1
            alpha = scale*np.cos(theta)
            beta = scale*np.sin(theta)
            rows,cols = gray.shape

            R = np.array([
                [alpha, beta, (1 - alpha) * eyes_centers[0][0] - beta * eyes_centers[0][1]],
                [-beta, alpha, beta * eyes_centers[0][0] + (1 - alpha) * eyes_centers[0][1]]
            ])
            rotated_gray = cv2.warpAffine(gray,R,(cols,rows))

            left_eye = np.array([eyes_centers[0][0],eyes_centers[0][1],1])
            right_eye = np.array([eyes_centers[1][0],eyes_centers[1][1],1])
            left_eye = R.dot(left_eye)
            right_eye = R.dot(right_eye)

            #Finding the correct caling 
            eyes_distance = np.linalg.norm(left_eye - right_eye)
            ratio = 15/eyes_distance

            #Scaling everything down
            scaled_height = int(np.rint(rows*ratio))
            scaled_width = int(np.rint(cols*ratio))
            left_eye = np.round(left_eye*ratio)
            right_eye = np.round(right_eye*ratio)
            scaled_gray = cv2.resize(rotated_gray, (scaled_width, scaled_height))
            # cv2.line(scaled_gray,(int(left_eye[0]),int(left_eye[1])),(int(right_eye[0]),int(right_eye[1])),(255,0,0), 2)
            print(scaled_gray.shape)
            print(left_eye[1]-22)
            print(left_eye[1]+24)
            print(left_eye[0]-16)
            print(right_eye[0]+15)

            cropped_gray = scaled_gray[int(left_eye[1]-24):int(left_eye[1]+32), int(left_eye[0]-16):int(right_eye[0]+15)]
            print(cropped_gray.shape)

            cv2.imwrite('myFaceDB/normalized/' + filename, cropped_gray)


