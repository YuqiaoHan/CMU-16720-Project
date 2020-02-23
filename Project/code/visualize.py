'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import analyze
import helper

im1 = plt.imread('../dino/dino0001.png')
im2 = plt.imread('../dino/dino0002.png')
M = max(im1.shape[0],im1.shape[1])
points = np.load('../results/matches.npz')
point1 = points['pts1']
point2 = points['pts2']
F = analyze.eightpoint(point1,point2,M)

x = np.load('../results/test_points.npz')
x1 = x['pts'][:,0]
y1 = x['pts'][:,1]

pts1 = np.transpose(np.stack((x1,y1),axis=0))

X2 = []
Y2 = []
for i in range(len(x1)):
    x2,y2 = analyze.epipolarCorrespondence(im1,im2,F,x1[i],y1[i])
    X2.append(x2)
    Y2.append(y2)
pts2 = np.transpose(np.stack((X2,Y2),axis=0))
print(pts2)

K1 = [[3310.400000, 0.000000, 316.730000],
      [0.000000, 3325.500000, 200.550000],
      [0.000000, 0.000000, 1.000000]] 
K2 = [[3310.400000, 0.000000, 316.730000],
      [0.000000, 3325.500000, 200.550000],
      [0.000000, 0.000000, 1.000000]]

E = analyze.essentialMatrix(F,K1,K2)
M1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
C1 = K1 @ M1
M2s = helper.camera2(E)
N = M2s.shape[2]

P_3d = None
error = None
M2 = None
C2 = None
for i in range(N):
    M2_tmp = M2s[:,:,i]
    C2_tmp = K2 @ M2_tmp
    w,err = analyze.triangulate(C1,pts1,C2_tmp,pts2)
    # P_3d = w
    # error = err
    # M2 = M2_tmp
    # C2 = C2_tmp
    if (np.min(w[:,-1])>0):
        P_3d = w
        error = err
        M2 = M2_tmp
        C2 = C2_tmp

wx = P_3d[:,0]
wy = P_3d[:,1]
wz = P_3d[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(wx, wy, wz, c='b', marker='.')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()