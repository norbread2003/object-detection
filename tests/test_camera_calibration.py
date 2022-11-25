import numpy as np
from scipy import linalg

from DLT import DLTcalib

obj_point = np.random.random((3, 6))

R_true = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, -1]
], np.float32)

t_true = np.array([
    [1],
    [-2],
    [3],
], np.float32)

K_true = np.array([
    [3, 0, 100],
    [0, 4, 200],
    [0, 0, 1]
])

img_point = np.matmul(K_true, np.matmul(R_true, obj_point) + t_true)

img_point = (img_point / img_point[2])[:2, :]

obj_point = np.array(
    [[0.00615378, 0.03448209, 0.00247359]
        , [0.00301123, 0.03489576, 0.00710771]
        , [-0.00838965, 0.03339605, -0.00250988]
        , [-0.01114489, 0.0322548, -0.00118637]
        , [-0.00860894, 0.03330516, 0.00026715]
        , [-0.01138934, 0.03215343, 0.00199963]]
    , np.float32).transpose()

img_point = np.array(
    [[952, 964]
        , [957, 983]
        , [912, 1004]
        , [913, 1016]
        , [920, 1011]
        , [921, 1023]]
    , np.float32).transpose()

init_cam_matrix, err = DLTcalib(3, obj_point.transpose(), img_point.transpose())
print("err", err)
init_cam_matrix = init_cam_matrix.reshape((3, 4))

A = init_cam_matrix[:3, :3]
K, R = linalg.rq(A)
t = linalg.solve(K, init_cam_matrix[:, 3])

# X = np.identity(3)
# for i in range(3):
#     if K[i, i] < 0:
#         K[:, i] = -K[:, i]
#         X[i, i] = -1
# R = np.matmul(X, R)
# t = np.matmul(X, t)

K /= K[2, 2]

print(K)
print(R)
print(t)
print(np.linalg.det(R))

img_point_pred = np.matmul(K, np.matmul(R, obj_point) + t.reshape((3, 1)))
img_point_pred /= img_point_pred[2:]
print(img_point_pred)
