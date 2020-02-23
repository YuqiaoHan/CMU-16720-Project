"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper
import cv2
import matplotlib.pyplot as plt
'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    x1 = pts1[:,0] / M
    y1 = pts1[:,1] / M
    x2 = pts2[:,0] / M
    y2 = pts2[:,1] / M
    T = np.array([[1/M,0,0],[0,1/M,0],[0,0,1]])
    A = []

    for i in range(len(x1)):
        A.append([x1[i]*x2[i],x1[i]*y2[i],x1[i],y1[i]*x2[i],y1[i]*y2[i],y1[i],x2[i],y2[i],1])

    u,s,v_h = np.linalg.svd(A)
    f = v_h[-1,:]
    F = f.reshape((3,3))

    F = helper.refineF(F,pts1/M,pts2/M)
    F = helper._singularize(F)

    F_unnorm = np.transpose(T) @ (F @ T)
    return F_unnorm

'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    x1 = pts1[:,0] / M
    y1 = pts1[:,1] / M
    x2 = pts2[:,0] / M
    y2 = pts2[:,1] / M
    T = np.array([[1/M,0,0],[0,1/M,0],[0,0,1]])
    A = []

    for i in range(len(x1)):
        A.append([x1[i]*x2[i],x1[i]*y2[i],x1[i],y1[i]*x2[i],y1[i]*y2[i],y1[i],x2[i],y2[i],1])

    u,s,v_h = np.linalg.svd(A)
    f1 = v_h[-1,:]
    f2 = v_h[-2,:]
    F1 = f1.reshape((3,3))
    F2 = f2.reshape((3,3))

    fun = lambda a: np.linalg.det(a * F1 + (1 - a) * F2)
    a0 = fun(0)
    a1 = 2*(fun(1)-fun(-1))/3 - (fun(2) - fun(-2))/12
    a2 = 0.5*fun(1) + 0.5*fun(-1) - fun(0)
    a3 = 0.5*fun(1) - 0.5*fun(-1) - a1
    a = np.roots([a3,a2,a1,a0])

    F = []
    for alpha in a:
        ff = alpha * F1 + (1 - alpha) * F2
        F.append(ff)

    F_refine = []
    for f in F:
        ref = helper.refineF(f, pts1/M, pts2/M)
        F_refine.append(ref)
    
    Farray = []
    for ref in F_refine:
        ref = np.transpose(T) @ (ref @ T)
        Farray.append(ref)
    Farray = np.asarray(Farray)
    return Farray
'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    E = np.transpose(K2) @ (F @ K1)
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]

    N = pts1.shape[0]
    w = np.zeros((N,3))
    for i in range(N):
        A1 = np.array([C1[2,0]*x1[i]-C1[0,0], C1[2,1]*x1[i]-C1[0,1], C1[2,2]*x1[i]-C1[0,2], C1[2,3]*x1[i]-C1[0,3]])
        A2 = np.array([C1[2,0]*y1[i]-C1[1,0], C1[2,1]*y1[i]-C1[1,1], C1[2,2]*y1[i]-C1[1,2], C1[2,3]*y1[i]-C1[1,3]])
        A3 = np.array([C2[2,0]*x2[i]-C2[0,0], C2[2,1]*x2[i]-C2[0,1], C2[2,2]*x2[i]-C2[0,2], C2[2,3]*x2[i]-C2[0,3]])
        A4 = np.array([C2[2,0]*y2[i]-C2[1,0], C2[2,1]*y2[i]-C2[1,1], C2[2,2]*y2[i]-C2[1,2], C2[2,3]*y2[i]-C2[1,3]])
        A = np.stack((A1,A2,A3,A4))
        u,s,v_h = np.linalg.svd(A)
        ww = v_h[-1,:]
        w[i,:] = ww[:3]/ww[3]
    
    one = np.ones((N,1))
    w_h = np.hstack((w,one))
    error = 0
    for j in range(N):
        proj1 = C1 @ w_h[j,:]
        proj1 = proj1[:2]/proj1[-1]
        proj2 = C2 @ w_h[j,:]
        proj2 = proj2[:2]/proj2[-1]
        error += np.sum((pts1[j] - proj1)**2 + (pts2[j] - proj2)**2)

    return w,error
'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    x1 = np.around(x1)
    y1 = np.around(y1)

    windowWidth = 13
    center = (int)(windowWidth/2)
    sigma = 6

    window = np.array([range(windowWidth)])
    window = np.repeat(window,windowWidth,axis=0) - center
    window = np.sqrt(window**2 + np.transpose(window)**2)
    gaussian_weight = np.exp(-1/2 * (window**2) / (sigma**2))
    gaussian_weight /= np.sum(gaussian_weight)

    if (len(im1.shape)>2):
        gaussian_weight = np.expand_dims(gaussian_weight, axis=2)
        gaussian_weight = np.repeat(gaussian_weight, im1.shape[-1], axis=2)
    
    pt1 = np.array([x1,y1,1])
    l2 = F @ pt1
    
    patch1 = im1[y1-center:y1+center+1,x1-center:x1+center+1]
    searchRange = 50
    y = np.array([range(y1-searchRange,y1+searchRange)])
    a = l2[0]
    b = l2[1]
    c = l2[2]
    x = np.around((-c - b*y)/a).astype(int)
    inrange_pos = []
    height = im2.shape[0]
    width = im2.shape[1]
    y = np.squeeze(y)
    x = np.squeeze(x)

    ind = 0
    for i,j in zip(x,y):
        if (i >= center) and (i < width - center) and (j >= center) and (j < height - center):
            inrange_pos.append(ind)
        ind += 1
    X2 = x[inrange_pos]
    Y2 = y[inrange_pos]

    min_dist = None
    x2 = None
    y2 = None
    for xx, yy in zip(X2,Y2):
        patch2 = im2[yy-center:yy+center+1, xx-center:xx+center+1]
        dist = np.sum((patch1-patch2)**2*gaussian_weight)
        if (min_dist == None) or (dist < min_dist):
            min_dist = dist
            x2 = xx
            y2 = yy

    return x2,y2
