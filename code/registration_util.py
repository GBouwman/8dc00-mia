"""
Utility functions for registration.
"""

import numpy as np
from cpselect.cpselect import cpselect


def test_object(centered=True):
    # Generate an F-like test object.
    # Input:
    # centered - set the object centroid to the origin
    # Output:
    # X - coordinates of the test object

    X = np.array([[4, 4, 4.5, 4.5, 6, 6, 4.5, 4.5, 7, 7, 4], [10, 4, 4, 7, 7, 7.5, 7.5, 9.5, 9.5, 10, 10]])

    if centered:
        X[0, :] = X[0, :] - np.mean(X[0, :])
        X[1, :] = X[1, :] - np.mean(X[1, :])

    return X


def c2h(X):
    # Convert cartesian to homogeneous coordinates.
    # Input:
    # X - cartesian coordinates
    # Output:
    # Xh - homogeneous coordinates

    n = np.ones([1,X.shape[1]])
    Xh = np.concatenate((X,n))

    return Xh


def t2h(T, t):
    # Convert a 2D transformation matrix to homogeneous form.
    # Input:
    # T - 2D transformation matrix
    # Xt - 2D translation vector
    # Output:
    # Th - homogeneous transformation matrix

    matrix = np.zeros((2,3))
    matrix[0][0] = T[0][0]
    matrix[0][1] = T[0][1]
    matrix[1][0] = T[1][0]
    matrix[1][1] = T[1][1]
    matrix[0][2] = t[0]
    matrix[1][2] = t[1]
    n = np.zeros([1,matrix.shape[1]])
    n[0][-1] = 1
    Th = np.concatenate((matrix,n), axis=0)

    return Th


def plot_object(ax, X):
    # Plot 2D object.

    # Input:
    # X - coordinates of the shape

    ax.plot(X[0,:], X[1,:], linewidth=2)


def my_cpselect(I_path, Im_path):
    # Wrapper around cpselect that returns the point coordinates
    # in the expected format (coordinates in rows).
    # Input:
    # I - fixed image
    # Im - moving image
    # Output:
    # X - control points in the fixed image
    # Xm - control points in the moving image
    #------------------------------------------------------------------#
    controlpointlist = cpselect(I_path,Im_path)
    #------------------------------------------------------------------#
    X_xcoor = []
    X_ycoor = []
    Xm_xcoor = []
    Xm_ycoor = []
    for ind in controlpointlist:
        X_xcoor.append(ind['img1_x'])
        X_ycoor.append(ind['img1_y'])
        Xm_xcoor.append(ind['img2_x'])
        Xm_ycoor.append(ind['img2_y'])
    X = np.array([X_xcoor,X_ycoor])
    Xm = np.array([Xm_xcoor,Xm_ycoor])
    return X, Xm
