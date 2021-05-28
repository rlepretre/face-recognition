import cv2
import numpy as np
import csv


def pca(x, mu, c):
    A = x - mu
    R = np.dot(A.T, A)

    eigenvalue, eigenvectors = np.linalg.eig(R)
    m = x.shape[1] - c
    index = eigenvalue.argsort()
    V = eigenvectors[:,index[-m:]]
    
    W = np.dot(A,V)
    norm = np.tile(np.linalg.norm(W, axis = 0),(W.shape[0],1))
    W = np.divide(W,norm)

    Y = np.dot(W.T,A)

    return A, W, Y

def reconstruction(W, Y, mu):
    X_reconstructed = np.dot(W,Y) + mu 
    img = np.abs(X_reconstructed[:,0].reshape(56,46))
    img = img/np.max(img)
    return img
