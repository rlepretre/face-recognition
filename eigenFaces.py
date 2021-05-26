import cv2
import numpy as np
import csv

x = np.empty((0,2576))

# reading csv file
with open('./faceDB_20_21/training/classes.csv', 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile, delimiter = ';')
    myClasses = []

    for row in csvreader:
        img = cv2.imread('./faceDB_20_21/training/' + row[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x_i = gray.flatten()
        x = np.vstack((x,x_i))
        myClasses = np.hstack((myClasses,int(row[1])))

x = x.T
mu = np.mat(np.mean(x, axis = 1)).T

A = x - mu

R = np.dot(A.T, A)
eigenvalue, eigenvectors = np.linalg.eig(R)

m = 83
index = eigenvalue.argsort()

V = eigenvectors[:,index[-m:]]
W = np.dot(A,V)
norm = np.tile(np.linalg.norm(W, axis = 0),(W.shape[0],1))
W = np.divide(W,norm)

Y = np.dot(W.T,A)

X_reconstructed = np.dot(W,Y) + mu 

img = np.abs(X_reconstructed[:,0].reshape(56,46))
img = img/np.max(img)

Y = Y.astype(np.float32)
myClasses = myClasses.astype(np.float32)



knn = cv2.ml.KNearest_create()
knn.train(Y, cv2.ml.COL_SAMPLE, myClasses)
ret, results, neighbours, dist = knn.findNearest(Y[:,14].T, 1)

print(results)
print(dist)