"""
Registration module main code.
"""

import numpy as np
from scipy import ndimage
import registration_util as util



# SECTION 1. Geometrical transformations


def identity():
    # 2D identity matrix.
    # Output:
    # T - transformation matrix

    T = np.eye(2)

    return T


def scale(sx, sy):
    # 2D scaling matrix.
    # Input:
    # sx, sy - scaling parameters
    # Output:
    # T - transformation matrix

    T = np.array([[sx,0],[0,sy]])

    return T


def rotate(phi):
    # 2D rotation matrix.
    # Input:
    # phi - rotation angle
    # Output:
    # T - transformation matrix

    T = np.array([[np.cos(phi),-(np.sin(phi))],[np.sin(phi),np.cos(phi)]])

    return T


def shear(cx, cy):
    # 2D shearing matrix.
    # Input:
    # cx - horizontal shear
    # cy - vertical shear
    # Output:
    # T - transformation matrix

    T = np.array([[1,cx],[cy,1]])

    return T


def reflect(rx, ry):
    # 2D reflection matrix.
    # Input:
    # rx - horizontal reflection (must have value of -1 or 1)
    # ry - vertical reflection (must have value of -1 or 1)
    # Output:
    # T - transformation matrix

    allowed = [-1, 1]
    if rx not in allowed or ry not in allowed:
        T = 'Invalid input parameter'
        return T

    else:
        T = np.array([[rx,0],[0,ry]])

    return T


# SECTION 2. Image transformation and least squares fitting


def image_transform(I, Th,  output_shape=None):
    # Image transformation by inverse mapping.
    # Input:
    # I - image to be transformed
    # Th - homogeneous transformation matrix
    # output_shape - size of the output image (default is same size as input)
    # Output:
    # It - transformed image
    # we want double precision for the interpolation, but we want the
    # output to have the same data type as the input - so, we will
    # convert to double and remember the original input type

    input_type = type(I);

    # default output size is same as input
    if output_shape is None:
        output_shape = I.shape

    # spatial coordinates of the transformed image
    x = np.arange(0, output_shape[1])
    y = np.arange(0, output_shape[0])
    xx, yy = np.meshgrid(x, y)

    # convert to a 2-by-p matrix (p is the number of pixels)
    X = np.concatenate((xx.reshape((1, xx.size)), yy.reshape((1, yy.size))))
    # convert to homogeneous coordinates
    Xh = util.c2h(X)
    #------------------------------------------------------------------#
    Rev_T = np.linalg.inv(Th)
    Xt = Rev_T.dot(Xh)
    #------------------------------------------------------------------#

    It = ndimage.map_coordinates(I, [Xt[1,:], Xt[0,:]], order=1, mode='constant').reshape(I.shape)

    return It, Xt


def ls_solve(A, b):
    # Least-squares solution to a linear system of equations.
    # Input:
    # A - matrix of known coefficients
    # b - vector of known constant term
    # Output:
    # w - least-squares solution to the system of equations
    # E - squared error for the optimal solution

    #------------------------------------------------------------------#
    w = (np.linalg.inv(np.transpose(A).dot(A))).dot(np.transpose(A)).dot(b)
    #------------------------------------------------------------------#

    # compute the error
    E = np.transpose(A.dot(w) - b).dot(A.dot(w) - b)

    return w, E


def ls_affine(X, Xm):
    # Least-squares fitting of an affine transformation.
    # Input:
    # X - Points in the fixed image
    # Xm - Corresponding points in the moving image
    # Output:
    # T - affine transformation in homogeneous form.

    A = np.transpose(Xm)

    bx = X[0]
    by = X[1]
    Tx, Ex = ls_solve(A,bx)
    Ty, Ey = ls_solve(A,by)
    T = np.array([Tx, Ty])


    return T


# SECTION 3. Image simmilarity metrics


def correlation(I, J):
    # Compute the normalized cross-correlation between two images.
    # Input:
    # I, J - input images
    # Output:
    # CC - normalized cross-correlation
    # it's always good to do some parameter checks

    if I.shape != J.shape:
        raise AssertionError("The inputs must be the same size.")

    u = I.reshape((I.shape[0]*I.shape[1],1))
    v = J.reshape((J.shape[0]*J.shape[1],1))

    # subtract the mean
    u = u - u.mean(keepdims=True)
    v = v - v.mean(keepdims=True)

    CC = (np.transpose(u).dot(v))/(np.sqrt(np.transpose(u).dot(u)).dot(np.sqrt(np.transpose(v).dot(v))))

    
    return CC


def joint_histogram(I, J, num_bins=16, minmax_range=None):
    # Compute the joint histogram of two signals.
    # Input:
    # I, J - input images
    # num_bins: number of bins of the joint histogram (default: 16)
    # range - range of the values of the signals (defaul: min and max
    # of the inputs)
    # Output:
    # p - joint histogram

    if I.shape != J.shape:
        raise AssertionError("The inputs must be the same size.")

    # make sure the inputs are column-vectors of type double (highest
    # precision)
    I = I.reshape((I.shape[0]*I.shape[1],1)).astype(float)
    J = J.reshape((J.shape[0]*J.shape[1],1)).astype(float)

    # if the range is not specified use the min and max values of the
    # inputs
    if minmax_range is None:
        minmax_range = np.array([min(min(I),min(J)), max(max(I),max(J))])

    # this will normalize the inputs to the [0 1] range
    I = (I-minmax_range[0]) / (minmax_range[1]-minmax_range[0])
    J = (J-minmax_range[0]) / (minmax_range[1]-minmax_range[0])

    # and this will make them integers in the [0 (num_bins-1)] range
    I = np.round(I*(num_bins-1)).astype(int)
    J = np.round(J*(num_bins-1)).astype(int)

    n = I.shape[0]
    hist_size = np.array([num_bins,num_bins])

    # initialize the joint histogram to all zeros
    p = np.zeros(hist_size)

    for k in range(n):
        p[I[k], J[k]] = p[I[k], J[k]] + 1

    p[I,J] = p[I,J]/len(I)

    return p


def mutual_information(p):
    # Compute the mutual information from a joint histogram.
    # Input:
    # p - joint histogram
    # Output:
    # MI - mutual information in nat units
    # a very small positive number

    EPSILON = 10e-10

    # add a small positive number to the joint histogram to avoid
    # numerical problems (such as division by zero)
    p += EPSILON

    # we can compute the marginal histograms from the joint histogram
    p_I = np.sum(p, axis=1)
    p_I = p_I.reshape(-1, 1)
    p_J = np.sum(p, axis=0)
    p_J = p_J.reshape(1, -1)
    
    n = len(p_I)
    MI = 0

    for i in range(n):
        for j in range(n):
            MI = MI + p[i, j]*np.log(p[i,j]/(p_I[i,0]*p_J[0,j]))

    return MI


def mutual_information_e(p):
    # Compute the mutual information from a joint histogram.
    # Alternative implementation via computation of entropy.
    # Input:
    # p - joint histogram
    # Output:
    # MI - mutual information in nat units
    # a very small positive number

    EPSILON = 10e-10

    # add a small positive number to the joint histogram to avoid
    # numerical problems (such as division by zero)
    p += EPSILON

    # we can compute the marginal histograms from the joint histogram
    p_I = np.sum(p, axis=1)
    p_I = p_I.reshape(-1, 1)
    p_J = np.sum(p, axis=0)
    p_J = p_J.reshape(1, -1)

    un = np.unique(p)
    un = un[1:]
    
    entropy1 = -sum(p_I*np.log(p_I))
    entropy2 = -sum(np.transpose(p_J)*np.log(np.transpose(p_J)))
    jointentropy = -sum(un*np.log(un))
    
    MI = entropy1 + entropy2 - jointentropy

    return MI


# SECTION 4. Towards intensity-based image registration


def ngradient(fun, x, h=1e-3):
    # Computes the derivative of a function with numerical differentiation.
    # Input:
    # fun - function for which the gradient is computed
    # x - vector of parameter values at which to compute the gradient
    # h - a small positive number used in the finite difference formula
    # Output:
    # g - vector of partial derivatives (gradient) of fun


    g = []
    if len(x) == 1:
        g = (fun(x+h/2)-fun(x-h/2))/h
    else:
        g = np.zeros(len(x))
        for k in range(len(x)):
            xp = x.copy()
            xn = x.copy()
            value = x[k]
            xp[k] = value + h/2
            xn[k] = value - h/2
            g[k] = (fun(xp)[0]-fun(xn)[0])/h
            
    return g


def rigid_corr(I, Im, x):
    # Computes normalized cross-correlation between a fixed and
    # a moving image transformed with a rigid transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the rigid transform: the first element
    #     is the rotation angle and the remaining two elements
    #     are the translation
    # Output:
    # C - normalized cross-correlation between I and T(Im)
    # Im_t - transformed moving image T(Im)

    SCALING = 100

    # the first element is the rotation angle
    T = rotate(x[0])

    # the remaining two element are the translation
    #
    # the gradient ascent/descent method work best when all parameters
    # of the function have approximately the same range of values
    # this is  not the case for the parameters of rigid registration
    # where the transformation matrix usually takes  much smaller
    # values compared to the translation vector this is why we pass a
    # scaled down version of the translation vector to this function
    # and then scale it up when computing the transformation matrix
    Th = util.t2h(T, x[1:]*SCALING)

    # transform the moving image
    Im_t, Xt = image_transform(Im, Th)

    # compute the similarity between the fixed and transformed
    # moving image
    C = correlation(I, Im_t)

    return C, Im_t, Th


def affine_corr(I, Im, x):
    # Computes normalized cross-corrleation between a fixed and
    # a moving image transformed with an affine transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the rigid transform: the first element
    #     is the roation angle, the second and third are the
    #     scaling parameters, the fourth and fifth are the
    #     shearing parameters and the remaining two elements
    #     are the translation
    # Output:
    # C - normalized cross-corrleation between I and T(Im)
    # Im_t - transformed moving image T(Im)

    NUM_BINS = 64
    SCALING = 100

    T_rot = rotate(x[0])
    T_scale = scale(x[1],x[2])
    T_shear = shear(x[3],x[4])
    
    T = T_rot.dot(T_scale).dot(T_shear)
    
    Th = util.t2h(T, x[5:]*SCALING)
    
    Im_t, Xt = image_transform(Im, Th)
    
    C = correlation(I, Im_t)

    return C, Im_t, Th


def affine_mi(I, Im, x):
    # Computes mutual information between a fixed and
    # a moving image transformed with an affine transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the rigid transform: the first element
    #     is the rotation angle, the second and third are the
    #     scaling parameters, the fourth and fifth are the
    #     shearing parameters and the remaining two elements
    #     are the translation
    # Output:
    # MI - mutual information between I and T(Im)
    # Im_t - transformed moving image T(Im)

    NUM_BINS = 64
    SCALING = 100
    
    T_rot = rotate(x[0])
    T_scale = scale(x[1],x[2])
    T_shear = shear(x[3],x[4])
    
    T = T_rot.dot(T_scale).dot(T_shear)
    
    Th = util.t2h(T, x[5:]*SCALING)
    
    Im_t, Xt = image_transform(Im, Th)
    
    p = joint_histogram(I,Im_t)
    
    MI = mutual_information(p)

    return MI, Im_t, Th
