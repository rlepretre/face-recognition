import cv2
from fisherFaces import fisherFaces

Y_train, myTrainingClasses  = fisherFaces('./faceDB_20_21/training/')
Y_test, myTestingClasses = fisherFaces('./faceDB_20_21/test/')


knn = cv2.ml.KNearest_create()
knn.train(Y_train, cv2.ml.ROW_SAMPLE, myTrainingClasses)
i = 0

for testVector in Y_test:
    ret, results, neighbours, dist = knn.findNearest(testVector, 1)
    correct = (results == myTestingClasses[i])
    print((results == myTestingClasses[i]))
    i = i + 1