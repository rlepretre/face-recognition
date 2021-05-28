import cv2
from fisherFaces import fisherFaces

Y_train, W_fld, W_pca, myTrainingClasses  = fisherFaces('./faceDB_20_21/training/')
Y_test, W_fld_test, W_pca_test, myTestingClasses = fisherFaces('./faceDB_20_21/test/')


knn = cv2.ml.KNearest_create()
knn.train(Y_train, cv2.ml.ROW_SAMPLE, myTrainingClasses)
i = 0

for testVector in Y_test:
    ret, results, neighbours, dist = knn.findNearest(testVector, 1)
    correct = (results == myTestingClasses[i])
    print((results == myTestingClasses[i]))
    # print(dist)
    i = i + 1