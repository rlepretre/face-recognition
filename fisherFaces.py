import cv2
import numpy as np
import csv
from eigenFaces import pca

def fisherFaces(training_set):

    x = np.empty((0,2576))

    # reads csv file and generates the class list
    with open(training_set + 'classes.csv', 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile, delimiter = ';')
        my_classes = []

        for row in csvreader:
            img = cv2.imread(training_set + row[0])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x_i = gray.flatten()
            x = np.vstack((x,x_i))
            my_classes = np.hstack((my_classes,int(row[1])))

    x = x.T
    my_classes_set = set(my_classes)

    class_means = np.empty((2576,len(my_classes_set)))
    S_w = np.empty((2576,2576))
    S_b = np.empty((2576,2576))
    

    #Determines the mean face mu
    mu = np.mat(np.mean(x, axis = 1)).T

    i = 0
    for c in range(len(my_classes_set)):
        class_amount = np.count_nonzero(my_classes == c+1)

        #Determines the mean face of each class mu_i
        class_means[:,c] = np.mat(np.mean(x[:,i:i+class_amount], axis = 1))
        
        #Tiles the mean vectors 
        class_mean = np.tile(class_means[:,c], (class_amount,1))
        global_mean_tiled = np.tile(mu, (1,class_amount)) 

        #Computes S_w and S_b
        S_w = S_w + np.dot((x[:,i:i+class_amount] - class_mean.T), (x[:,i:i+class_amount] - class_mean.T).T)
        S_b = S_b + class_amount*np.dot((class_mean - global_mean_tiled.T).T,(class_mean - global_mean_tiled.T))

        #Increment i by n_i (the amount of x_i per class)
        i = i + class_amount

    #Computes W_pca with the eigenfaces algorithm
    A , W_pca, Y_pca = pca(x, mu, len(my_classes_set))

    #Computes S_b and S_w with the curves on top (W_pca.T * S_b * W_pca) and 
    S_b_curve = np.linalg.multi_dot([W_pca.T, S_b, W_pca]) 
    S_w_curve = np.linalg.multi_dot([W_pca.T, S_w, W_pca]) 
    final_matrix = np.dot(np.linalg.inv(S_w_curve), S_b_curve)

    #Determine the c-1 “larger eigenvectors” from the final matrix
    larger_eigenvalue, larger_eigenvectors = np.linalg.eig(final_matrix)
    m = len(my_classes_set) - 1
    index = larger_eigenvalue.argsort()
    W_fld = larger_eigenvectors[:,index[-m:]]

    #Find Y_fld
    Y_fld = np.linalg.multi_dot([W_fld.T.real, W_pca.T, A]) 

    Y_fld = Y_fld.astype(np.float32)
    my_classes = my_classes.astype(np.float32)

    return Y_fld.T, W_pca, W_fld, my_classes


