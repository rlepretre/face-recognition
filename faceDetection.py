# import numpy as np
import cv2
import numpy as np
from helpers import normalize, addOverlay
from fisherFaces import fisherFaces

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

Y_train, W_pca, W_fld, my_classes, mu  = fisherFaces('./myFaceDB/normalized/')
knn = cv2.ml.KNearest_create()
knn.train(Y_train, cv2.ml.ROW_SAMPLE, my_classes)


cv2.namedWindow("Mac Camera")
vc = cv2.VideoCapture(0)
if vc.isOpened(): #pauses to make sure the frame can be read
    rval, frame = vc.read()
else:
    rval = False

while rval:
    results = 0
    dist = 0
    rval, frame = vc.read()
    img, left_eye, theta, eyes_dist = normalize(frame)

    if img is not None and img.shape == (56,46):
        img = img.reshape(2576,1).astype(np.float32)


        W_fld = W_fld.real
        projected_img = np.linalg.multi_dot([W_fld.T, W_pca.T, img - mu]) 

        projected_img = projected_img.astype(np.float32)

        ret, results, neighbours, dist = knn.findNearest(projected_img.T, 1)

    print(dist) 

    if dist <= 9999999:
    
        if results == 4:
            virtual_object = cv2.imread('assets/glasses.jpg')
            mask_left_eye = np.array([240, 240, 1])
            mask_eyes_dist = 500
        
        if results == 3:
            virtual_object = cv2.imread('assets/fox_mask.jpg')
            mask_left_eye = np.array([435, 635, 1])
            mask_eyes_dist = 435

        if results == 2:
            virtual_object = cv2.imread('assets/wolf_mask.jpg')
            mask_left_eye = np.array([440, 530, 1])
            mask_eyes_dist = 330

        if results == 1:
            virtual_object = cv2.imread('assets/batman_mask.jpg')
            mask_left_eye = np.array([290, 720, 1])
            mask_eyes_dist = 300
    else: 
        virtual_object = cv2.imread('assets/zorro_mask.jpg')
        mask_left_eye = np.array([240, 220, 1])
        mask_eyes_dist = 420

    


    
    
    if(eyes_dist != 0):
        ratio = eyes_dist / mask_eyes_dist
        ar_img = addOverlay(frame, virtual_object, left_eye, mask_left_eye, ratio, -theta)
        cv2.imshow("Mac Camera", ar_img)
    else:
        cv2.imshow("Mac Camera", frame)
    

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC key
        break

    cv2.destroyWindow("Mac Camera")


