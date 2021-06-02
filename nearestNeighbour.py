import cv2
from fisherFaces import fisherFaces
import csv
import numpy as np

Y_train, W_pca, W_fld,  myTrainingClasses, mu  = fisherFaces('./faceDB_20_21/training/')

x = np.empty((0,2576))
with open('./faceDB_20_21/test/' + 'classes.csv', 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile, delimiter = ';')
        my_classes = []

        for row in csvreader:
            img = cv2.imread('./faceDB_20_21/test/' + row[0])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x_i = gray.flatten()
            x = np.vstack((x,x_i))
            my_classes = np.hstack((my_classes,int(row[1])))

x = x.T

Y_test = np.linalg.multi_dot([W_fld.T, W_pca.T, x - mu]).T 
knn = cv2.ml.KNearest_create()
knn.train(Y_train, cv2.ml.ROW_SAMPLE, myTrainingClasses)
i = 0
Y_test = Y_test.astype(np.float32)

ret, results, neighbours, dist = knn.findNearest(Y_test, 1)
count = np.count_nonzero((results.T == my_classes) == True)
print(count/len(my_classes))