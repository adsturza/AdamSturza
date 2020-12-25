import os
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

def load_dataset():

    root_path = os.path.dirname(os.path.abspath(__file__))
    
    train_path = os.path.join(root_path, 'train_catvnoncat.h5')
    train_dataset = h5py.File(train_path, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # train set labels

    test_path = os.path.join(root_path, 'test_catvnoncat.h5')
    test_dataset = h5py.File(test_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # test set labels

    classes = np.array(test_dataset["list_classes"][:]) # list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def pre(train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes):
    
    m_train = train_set_x_orig.shape[0]
    #test_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
    
    #print ("Number of training examples: m_train = " + str(m_train))
    #print ("Number of testing examples: m_test = " + str(m_test))
    #print ("Height/Width of each image: num_px = " + str(num_px))
    #print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    #print ("train_set_x shape: " + str(train_set_x_orig.shape))
    #print ("train_set_y shape: " + str(train_set_y.shape))
    #print ("test_set_x shape: " + str(test_set_x_orig.shape))
    #print ("test_set_y shape: " + str(test_set_y.shape))

    # flatten dimensions
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T   
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    #print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    #print ("train_set_y shape: " + str(train_set_y.shape))
    #print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    #print ("test_set_y shape: " + str(test_set_y.shape))
    #print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

    # normalize
    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.

    return train_set_x, test_set_x

def sigmoid(z):

    s = 1 / (1 + (np.exp(-z)))
    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    return w, b

def propagate(w, b, X, Y):

    m = X.shape[1]

    # Forward prop
    A = sigmoid(np.dot(w.T, X) + b)                       # activation
    cost = (-1/m)*(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))) # cost

    # Backward prop
    dw = (1 / m) * np.dot(X, (A - Y).T)                   # gradient of the loss with respect to w  
    db = (1 / m) * np.sum(A - Y)                          # gradient of the loss with respect to b

    cost = np.squeeze(cost)

    grads = {"dw": dw,
             "db": db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        
        # Derivatives
        dw = grads["dw"]
        db = grads["db"]
        
        # Update
        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)
        
        # Append costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T,X) + b)  
    
    for i in range(A.shape[1]):
        Y_prediction = (A >= 0.5) * 1.0
    
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):

    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Get parameters w and b
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

def visualize_learning_rate(d):
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()

def main():

    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    train_set_x, test_set_x = pre(train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes)

    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

    visualize_learning_rate(d)
    
if __name__ == '__main__':

    main()