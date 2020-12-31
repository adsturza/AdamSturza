import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
import scipy
from PIL import Image
from scipy import ndimage
from helper_utils import *
import os

def pre(train_x_orig, train_y, test_x_orig, test_y, classes):
    #m_train = train_x_orig.shape[0]
    #num_px = train_x_orig.shape[1]
    #m_test = test_x_orig.shape[0]

    #print ("Number of training examples: " + str(m_train))
    #print ("Number of testing examples: " + str(m_test))
    #print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    #print ("train_x_orig shape: " + str(train_x_orig.shape))
    #print ("train_y shape: " + str(train_y.shape))
    #print ("test_x_orig shape: " + str(test_x_orig.shape))
    #print ("test_y shape: " + str(test_y.shape))

    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T  
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    #print ("train_x's shape: " + str(train_x.shape))
    #print ("test_x's shape: " + str(test_x.shape))

    # constants defining model
    n_x = 12288
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)

    return train_x, test_x, layers_dims

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    
    # Two layer neural network: Linear -> RELU -> Linear -> Sigmoid

    grads = {}
    costs = []                              
    m = X.shape[1]                           
    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):

        A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")

        cost = compute_cost(A2, Y)

        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

        return parameters, costs, learning_rate

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):

    # L layer neural network: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID

    costs = []
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):

        # Forward prop
        AL, caches = L_model_forward(X, parameters)

        # Costs
        cost = compute_cost(AL, Y)

        # Backward prop
        grads = L_model_backward(AL, Y, caches)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    return parameters, costs, learning_rate

def visualize_learning_rate(c, l_r):

    plt.plot(np.squeeze(c))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(l_r))
    plt.show()

def main():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    train_x, test_x, layers_dims = pre(train_x_orig, train_y, test_x_orig, test_y, classes)

    parameters, costs, learning_rate = two_layer_model(train_x, train_y, layers_dims=layers_dims, num_iterations=2500, print_cost=True)
    visualize_learning_rate(costs, learning_rate)

    predictions_train = predict(train_x, train_y, parameters) # 100% in-sample accuracy
    predictions_test = predict(test_x, test_y, parameters) # ~ 80% out-of-sample accuracy

    parameters, costs, learning_rate = L_layer_model(train_x, train_y, layers_dims=[12288, 20, 7, 5, 1], num_iterations=2500, print_cost=True)
    visualize_learning_rate(costs, learning_rate)

    pred_train = predict(train_x, train_y, parameters) # 99% in-sample accuracy
    pred_test = predict(test_x, test_y, parameters) # ~ 82% out-of-sample accuracy

if __name__ == '__main__':
    main()