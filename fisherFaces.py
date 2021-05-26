import cv2
import numpy as np
import csv

def fisherFaces(trainingSet):

    x = np.empty((0,2576))

    # reads csv file and generates the class list
    with open(trainingSet + 'classes.csv', 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile, delimiter = ';')
        myClasses = []

        for row in csvreader:
            img = cv2.imread(trainingSet + row[0])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x_i = gray.flatten()
            x = np.vstack((x,x_i))
            myClasses = np.hstack((myClasses,int(row[1])))

    x = x.T
    myClassesSet = set(myClasses)

    classMeans = np.empty((2576,len(myClassesSet)))
    S_w = np.empty((2576,2576))
    S_b = np.empty((2576,2576))
    i = 0




    #Determines the mean face mu
    
    mu = np.mat(np.mean(x, axis = 1)).T


    for c in range(len(myClassesSet)):
        classAmount = np.count_nonzero(myClasses == c+1)

        #Determines the mean face of each class mu_i
        classMeans[:,c] = np.mat(np.mean(x[:,i:i+classAmount], axis = 1))
        
        #Tiles the mean vectors 
        classMean = np.tile(classMeans[:,c], (classAmount,1))
        globalMeanTiled = np.tile(mu, (1,classAmount)) 

        #Computes S_w and S_b
        S_w = S_w + np.dot((x[:,i:i+classAmount] - classMean.T), (x[:,i:i+classAmount] - classMean.T).T)
        S_b = S_b + classAmount*np.dot((classMean - globalMeanTiled.T).T,(classMean - globalMeanTiled.T))

        #Increment i by n_i (the amount of x_i per class)
        i = i + classAmount

    #Computes W_pca with the eigenfaces algorithm
    A = x - mu
    R = np.dot(A.T, A)
    eigenvalue, eigenvectors = np.linalg.eig(R)
    m = x.shape[1] - len(myClassesSet)
    index = eigenvalue.argsort()
    V = eigenvectors[:,index[-m:]]
    W_pca = np.dot(A,V)
    norm = np.tile(np.linalg.norm(W_pca, axis = 0),(W_pca.shape[0],1))
    W_pca = np.divide(W_pca,norm)

    #Computes S_b and S_w with the curves on top (W_pca.T * S_b * W_pca) and 
    S_b_curve = np.linalg.multi_dot([W_pca.T, S_b, W_pca]) 
    S_w_curve = np.linalg.multi_dot([W_pca.T, S_w, W_pca]) 
    finalMatrix = np.dot(np.linalg.inv(S_w_curve), S_b_curve)

    #Determine the c-1 “larger eigenvectors” from the final matrix
    largerEigenvalue, largerEigenvectors = np.linalg.eig(finalMatrix)
    index = largerEigenvalue.argsort()
    m = len(myClassesSet) - 1

    W_fld = largerEigenvectors[:,index[-m:]]

    #Find Y_fld
    Y_fld = np.linalg.multi_dot([W_fld.T, W_pca.T, A]) 

    Y_fld = Y_fld.astype(np.float32)
    myClasses = myClasses.astype(np.float32)

    return Y_fld.T, myClasses


