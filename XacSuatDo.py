# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
               2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# gắn 1 số vào sau X
X = X.T
y = y.T
# extended data
one = np.ones((X.shape[0],1))
X = np.concatenate((one, X), axis=1)


def sigmoid(s):
    return 1 / (1 + np.exp(-s))*100


def gradient_descent(X, y, theta_init, eta=0.05):
    theta_old = theta_init
    theta_epoch = theta_init

    N = X.shape[0]
    for it in range(10000):
        for i in range(N):
            xi = X[i, :]
            yi = y[i]
            zi = 1.0 / (1 + np.exp(-np.dot(xi, theta_old.T)))
            gi = (zi - yi) * xi

            theta_new = theta_old - eta*gi
            theta_old = theta_new

            if np.linalg.norm(theta_epoch - theta_new) < 1e-3:
                break
        theta_epoch = theta_old

    return theta_epoch, it


theta_init = np.array([1.5, 0.5])
print(theta_init)
theta, it = gradient_descent(X, y, theta_init)

hour = float(input('Nhập vào số giờ: '))
X = theta[1] * hour + theta[0]
print('% pass là', str(sigmoid(X)) + '%')
