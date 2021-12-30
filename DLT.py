# Marcos Duarte
# http://lob.iv.fapesp.br/
# University of Sao Paulo, Brazil
'''
Camera calibration and point reconstruction based on direct linear transformation (DLT).

The fundamental problem here is to find a mathematical relationship between the
 coordinates  of a 3D point and its projection onto the image plane. The DLT
 (a linear apporximation to this problem) is derived from modelling the object
 and its projection on the image plane as a pinhole camera situation.
In simplistic terms, using the pinhole camera model, it can be found by similar
 triangles the following relation between the image coordinates (u,v) and the 3D
 point (X,Y,Z):
   [ u ]   [ L1  L2  L3  L4 ] [ X ]
   [ v ] = [ L5  L6  L7  L8 ] [ Y ]
   [ 1 ]   [ L9 L10 L11 L12 ] [ Z ]
                              [ 1 ]
The matrix L is kwnown as the camera matrix or camera projection matrix. For a
 2D point (X,Y), the last column of the matrix doesn't exist. In fact, the L12
 term (or L9 for 2D DLT) is not independent from the other parameters and then
 there are only 11 (or 8 for 2D DLT) independent parameters in the DLT to be
 determined.

DLT is typically used in two steps: 1. camera calibration and 2. object (point)
 reconstruction.
The camera calibration step consists in digitizing points with known coordiantes
 in the real space.
At least 4 points are necessary for the calibration of a plane (2D DLT) and at
 least 6 points for the calibration of a volume (3D DLT). For the 2D DLT, at least
 one view of the object (points) must be entered. For the 3D DLT, at least 2
 different views of the object (points) must be entered.
These coordinates (from the object and image(s)) are inputed to the DLTcalib
 algorithm which  estimates the camera parameters (8 for 2D DLT and 11 for 3D DLT).
With these camera parameters and with the camera(s) at the same position of the
 calibration step,  we now can reconstruct the real position of any point inside
 the calibrated space (area for 2D DLT and volume for the 3D DLT) from the point
 position(s) viewed by the same fixed camera(s).

This code can perform 2D or 3D DLT with any number of views (cameras).
For 3D DLT, at least two views (cameras) are necessary.

There are more accurate (but more complex) algorithms for camera calibration that
 also consider lens distortion. For example, OpenCV and Tsai softwares have been
 ported to Python. However, DLT is classic, simple, and effective (fast) for
 most applications.

About DLT, see: http://kwon3d.com/theory/dlt/dlt.html

This code is based on different implementations and teaching material on DLT
 found in the internet.
'''

# Marcos Duarte - [EMAIL PROTECTED] - 04dec08

import numpy as N


def DLTcalib(nd, xyz, uv):
    '''
    Camera calibration by DLT using known object points and their image points.

    This code performs 2D or 3D DLT camera calibration with any number of views (cameras).
    For 3D DLT, at least two views (cameras) are necessary.
    Inputs:
     nd is the number of dimensions of the object space: 3 for 3D DLT and 2 for 2D DLT.
     xyz are the coordinates in the object 3D or 2D space of the calibration points.
     uv are the coordinates in the image 2D space of these calibration points.
     The coordinates (x,y,z and u,v) are given as columns and the different points as rows.
     For the 2D DLT (object planar space), only the first 2 columns (x and y) are used.
     There must be at least 6 calibration points for the 3D DLT and 4 for the 2D DLT.
    Outputs:
     L: array of the 8 or 11 parameters of the calibration matrix
     err: error of the DLT (mean residual of the DLT transformation in units of camera coordinates).
    '''

    # Convert all variables to numpy array:
    xyz = N.asarray(xyz)
    uv = N.asarray(uv)
    # number of points:
    np = xyz.shape[0]
    # Check the parameters:
    if uv.shape[0] != np:
        raise ValueError('xyz (%d points) and uv (%d points) have different number of points.' % (np, uv.shape[0]))
    if (nd == 2 and xyz.shape[1] != 2) or (nd == 3 and xyz.shape[1] != 3):
        raise ValueError('Incorrect number of coordinates (%d) for %dD DLT (it should be %d).' % (xyz.shape[1], nd, nd))
    if nd == 3 and np < 6 or nd == 2 and np < 4:
        raise ValueError('%dD DLT requires at least %d calibration points. Only %d points were entered.' % (
            nd, 2 * nd, np))

    # Normalize the data to improve the DLT quality (DLT is dependent of the system of coordinates).
    # This is relevant when there is a considerable perspective distortion.
    # Normalization: mean position at origin and mean distance equals to 1 at each direction.
    Txyz, xyzn = Normalization(nd, xyz)
    Tuv, uvn = Normalization(2, uv)

    A = []
    if nd == 2:  # 2D DLT
        for i in range(np):
            x, y = xyzn[i, 0], xyzn[i, 1]
            u, v = uvn[i, 0], uvn[i, 1]
            A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
            A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    elif nd == 3:  # 3D DLT
        for i in range(np):
            x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
            u, v = uvn[i, 0], uvn[i, 1]
            A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
            A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])

    # convert A to array
    A = N.asarray(A)
    # Find the 11 (or 8 for 2D DLT) parameters:
    U, S, Vh = N.linalg.svd(A)
    # The parameters are in the last line of Vh and normalize them:
    L = Vh[-1, :] / Vh[-1, -1]
    # Camera projection matrix:
    H = L.reshape(3, nd + 1)
    # Denormalization:
    H = N.dot(N.dot(N.linalg.pinv(Tuv), H), Txyz);
    H = H / H[-1, -1]
    L = H.flatten()
    # Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
    uv2 = N.dot(H, N.concatenate((xyz.T, N.ones((1, xyz.shape[0])))))
    uv2 = uv2 / uv2[2, :]
    # mean distance:
    err = N.sqrt(N.mean(N.sum((uv2[0:2, :].T - uv) ** 2, 1)))

    return L, err


def DLTrecon(nd, nc, Ls, uvs):
    '''
    Reconstruction of object point from image point(s) based on the DLT parameters.

    This code performs 2D or 3D DLT point reconstruction with any number of views (cameras).
    For 3D DLT, at least two views (cameras) are necessary.
    Inputs:
     nd is the number of dimensions of the object space: 3 for 3D DLT and 2 for 2D DLT.
     nc is the number of cameras (views) used.
     Ls (array type) are the camera calibration parameters of each camera
      (is the output of DLTcalib function). The Ls parameters are given as columns
      and the Ls for different cameras as rows.
     uvs are the coordinates of the point in the image 2D space of each camera.
      The coordinates of the point are given as columns and the different views as rows.
    Outputs:
     xyz: point coordinates in space
    '''

    # Convert Ls to array:
    Ls = N.asarray(Ls)
    # Check the parameters:
    if Ls.ndim == 1 and nc != 1:
        raise ValueError(
            'Number of views (%d) and number of sets of camera calibration parameters (1) are different.' % (
                nc))
    if Ls.ndim > 1 and nc != Ls.shape[0]:
        raise ValueError(
            'Number of views (%d) and number of sets of camera calibration parameters (%d) are different.' % (
                nc, Ls.shape[0]))
    if nd == 3 and Ls.ndim == 1:
        raise ValueError('At least two sets of camera calibration parameters are needed for 3D point reconstruction.')

    if nc == 1:  # 2D and 1 camera (view), the simplest (and fastest) case
        # One could calculate inv(H) and input that to the code to speed up things if needed.
        # (If there is only 1 camera, this transformation is all Floatcanvas2 might need)
        Hinv = N.linalg.inv(Ls.reshape(3, 3))
        # Point coordinates in space:
        xyz = N.dot(Hinv, [uvs[0], uvs[1], 1])
        xyz = xyz[0:2] / xyz[2]
    else:
        M = []
        for i in range(nc):
            L = Ls[i, :]
            u, v = uvs[i][0], uvs[i][1]  # this indexing works for both list and numpy array
            if nd == 2:
                M.append([L[0] - u * L[6], L[1] - u * L[7], L[2] - u * L[8]])
                M.append([L[3] - v * L[6], L[4] - v * L[7], L[5] - v * L[8]])
            elif nd == 3:
                M.append([L[0] - u * L[8], L[1] - u * L[9], L[2] - u * L[10], L[3] - u * L[11]])
                M.append([L[4] - v * L[8], L[5] - v * L[9], L[6] - v * L[10], L[7] - v * L[11]])

        # Find the xyz coordinates:
        U, S, Vh = N.linalg.svd(N.asarray(M))
        # Point coordinates in space:
        xyz = Vh[-1, 0:-1] / Vh[-1, -1]

    return xyz


def Normalization(nd, x):
    '''
    Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3).

    Inputs:
     nd: number of dimensions (2 for 2D; 3 for 3D)
     x: the data to be normalized (directions at different columns and points at rows)
    Outputs:
     Tr: the transformation matrix (translation plus scaling)
     x: the transformed data
    '''

    x = N.asarray(x)
    m, s = N.mean(x, 0), N.std(x)
    if nd == 2:
        Tr = N.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        Tr = N.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])

    Tr = N.linalg.inv(Tr)
    x = N.dot(Tr, N.concatenate((x.T, N.ones((1, x.shape[0])))))
    x = x[0:nd, :].T

    return Tr, x
