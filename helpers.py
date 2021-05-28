import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def normalize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.4, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3)
        eyes_centers = []

        for (ex,ey,ew,eh) in eyes:
            eyes_centers.append(np.array((x+(ex + ew//2),y + (ey + eh//2))))
        
        if len(eyes_centers) >= 2 and eyes_centers is not None:
            eye_one = np.array([eyes_centers[0][0],eyes_centers[0][1],1])
            eye_two = np.array([eyes_centers[1][0],eyes_centers[1][1],1])

            if(eye_one[0] < eye_two[0]):
                left_eye = eye_one
                original_left_eye = eye_one
                right_eye = eye_two
            else:
                left_eye = eye_two
                original_left_eye = eye_two
                right_eye = eye_one

            hyp = np.linalg.norm(left_eye-right_eye)
            theta = -np.arcsin((left_eye[1]-right_eye[1])/hyp)

            scale = 1
            alpha = scale*np.cos(theta)
            beta = scale*np.sin(theta)
            rows,cols = gray.shape

            R = np.array([
                [alpha, beta, (1 - alpha) * left_eye[0] - beta * left_eye[1]],
                [-beta, alpha, beta * left_eye[0] + (1 - alpha) * left_eye[1]]
            ])
            rotated_gray = cv2.warpAffine(gray,R,(cols,rows))

            left_eye = R.dot(left_eye)
            right_eye = R.dot(right_eye)

            #Finding the correct scaling 
            eyes_distance = np.linalg.norm(right_eye - left_eye)
            ratio = 15/eyes_distance

            #Scaling everything down
            scaled_height = int(np.rint(rows*ratio))
            scaled_width = int(np.rint(cols*ratio))
            left_eye = np.round(left_eye*ratio)
            right_eye = np.round(right_eye*ratio)
            scaled_gray = cv2.resize(rotated_gray, (scaled_width, scaled_height))
            # cv2.line(scaled_gray,(int(left_eye[0]),int(left_eye[1])),(int(right_eye[0]),int(right_eye[1])),(255,0,0), 2)

            normalized_img = scaled_gray[int(left_eye[1]-24):int(left_eye[1]+32), int(left_eye[0]-16):int(right_eye[0]+15)]

            return normalized_img, original_left_eye, theta, eyes_distance

    return None, [], 0, 0

def addOverlay(img, virtual_object, left_eye, mask_left_eye, ratio, angle):

    mask_left_eye = (mask_left_eye*ratio).astype(np.int32)
    print(left_eye)
    print(mask_left_eye)
    scaled_height = int(virtual_object.shape[1] * ratio) 
    scaled_width = int(virtual_object.shape[0] * ratio) 
    virtual_object = cv2.resize(virtual_object, (scaled_height,scaled_width), interpolation = cv2.INTER_AREA)

    y = int(left_eye[1] - mask_left_eye[1])
    x = int(left_eye[0] - mask_left_eye[0])

    background_width = img.shape[1]
    background_height = img.shape[0]

    h, w = virtual_object.shape[0], virtual_object.shape[1]

    lower_green = np.array([0, 100, 0])     ##[R value, G value, B value]
    upper_green = np.array([120, 255, 100]) 
    mask = cv2.inRange(virtual_object, lower_green, upper_green)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    inv_mask = np.bitwise_not(mask)
    mask = mask*255
    inv_mask = inv_mask*255
    

    img[y:y+h, x:x+w] = inv_mask * virtual_object + img[y:y+h, x:x+w] * mask

    return img

    