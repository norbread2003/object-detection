import math

import PIL.Image
import cv2
import numpy as np
from scipy import linalg

from DLT import DLTcalib


def draw_center_indicator(im: PIL.Image, center_indicator_length=0.2, center_indicator_thick=0.03):
    w = im.size[0] // 2
    h = im.size[1] // 2

    l = int(im.size[0] * center_indicator_length // 2)
    t = int(im.size[0] * center_indicator_thick // 2)

    upper_left_w: int = w - l
    upper_left_h: int = h - t

    bottom_right_w: int = w + l
    bottom_right_h: int = h + t

    im = _invert_color_in_box(im, upper_left_w, upper_left_h, bottom_right_w, bottom_right_h)

    upper_left_w: int = w - t
    upper_left_h: int = h - l

    bottom_right_w: int = w + t
    bottom_right_h: int = h + l

    im = _invert_color_in_box(im, upper_left_w, upper_left_h, bottom_right_w, bottom_right_h)

    return im


def _invert_color_in_box(im, upper_left_w, upper_left_h, bottom_right_w, bottom_right_h):
    is_rgb = type(im.getpixel((0, 0))) == tuple
    if is_rgb:
        new_value = [0, 0, 0]
    else:
        new_value = 0
    for i in range(upper_left_w, bottom_right_w + 1):
        for j in range(upper_left_h, bottom_right_h + 1):
            value = im.getpixel((i, j))
            if is_rgb:
                for k in range(3):
                    new_value[k] = 255 - value[k]
                im.putpixel((i, j), tuple(new_value))
            else:
                new_value = 255 - value
                im.putpixel((i, j), new_value)

    return im


def map_center_to_3d_point(center: tuple, depth_map):
    w, h = center
    point = [0, 0, 0]
    point[0] = w
    point[1] = h
    point[2] = depth_map[h][w]
    return point


def quaternion_rotating_around_x_axis(degree):
    return [math.cos(1 / 2 * math.pi * degree / 180), - math.sin(1 / 2 * math.pi * degree / 180), 0, 0]


def quaternion_rotating_around_y_axis(degree):
    return [math.cos(1 / 2 * math.pi * degree / 180), 0, - math.sin(1 / 2 * math.pi * degree / 180), 0]


def quaternion_rotating_around_z_axis(degree):
    return [math.cos(1 / 2 * math.pi * degree / 180), 0, 0, - math.sin(1 / 2 * math.pi * degree / 180)]


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


def get_camera_intrinsic_matrix(physics, camera_name, image_width, image_height) -> np.ndarray:
    cam_id = physics.model.name2id(camera_name, "camera")
    fovy = physics.model.cam_fovy[cam_id]
    f = 0.5 * image_height / math.tan(fovy * math.pi / 360)
    K = np.array(((f, 0, image_width / 2), (0, f, image_height / 2), (0, 0, 1)))

    return K


def get_camera_frame_wrt_world(physics, camera_name):
    cam_id = physics.model.name2id(camera_name, "camera")
    t = physics.model.cam_pos0[cam_id]
    R = physics.model.cam_mat0[cam_id].reshape((3, 3)).copy()

    R[:, 1] = -R[:, 1]
    R[:, 2] = -R[:, 2]

    return R, t


def uvh_to_X_cam(u, v, h, K) -> np.ndarray:
    z = h
    px = K[0, 2]
    py = K[1, 2]

    x = (u * z - z * px) / K[0, 0]
    y = (v * z - z * py) / K[1, 1]

    return np.array([x, y, z])


def X_cam_to_uv(X_cam, K) -> np.array:
    uv = K.dot(X_cam)
    uv /= uv[2]
    uv = uv[:2]

    return uv


def _test_convert_uvh_to_xyz():
    from dm_control import mujoco
    w = 200
    h = 100
    p = mujoco.Physics.from_xml_path("mujoco_models/world_0.xml")
    K = get_camera_intrinsic_matrix(p, "cam_0", w, h)
    R, t = get_camera_frame_wrt_world(p, "cam_0")

    X_cam_upper_left = uvh_to_X_cam(0, 0, t[2] / 2, K)
    X_world_upper_left = np.matmul(R, X_cam_upper_left) + t

    X_cam_bottom_left = uvh_to_X_cam(0, h, t[2] / 2, K)
    X_world_bottom_left = np.matmul(R, X_cam_bottom_left) + t

    X_cam_upper_right = uvh_to_X_cam(w, 0, t[2] / 2, K)
    X_world_upper_right = np.matmul(R, X_cam_upper_right) + t

    X_cam_bottom_right = uvh_to_X_cam(w, h, t[2] / 2, K)
    X_world_bottom_right = np.matmul(R, X_cam_bottom_right) + t


def draw_obj_vertices(plot, obj_path):
    v = []
    obj = open(obj_path, 'r')
    for line in obj:
        arr = line.split()
        if len(arr) == 0:
            continue

        if arr[0] == 'v':
            assert len(arr) == 4
            v.append([float(arr[i]) for i in range(1, 4)])

    v = np.array(v)
    a = 1.5 * v.min()
    b = 1.5 * v.max()

    plot.set_xlim3d(a, b)
    plot.set_ylim3d(a, b)
    plot.set_zlim3d(a, b)

    plot.quiver(0, 0, 0, 1, 0, 0, length=(b - a) / 3,
                arrow_length_ratio=0.1, color="red")
    plot.quiver(0, 0, 0, 0, 1, 0, length=(b - a) / 3,
                arrow_length_ratio=0.1, color="green")
    plot.quiver(0, 0, 0, 0, 0, 1, length=(b - a) / 3,
                arrow_length_ratio=0.1, color="blue")

    xs = v[:, 0]
    ys = v[:, 1]
    zs = v[:, 2]
    plot.scatter(xs, ys, zs, c="gray", alpha=0.1, linewidths=0.1)


def estimate_cam_K_R_t_opencvsolvePnP(img_point: np.array, obj_point: np.array, K: np.array):
    retval, rvec, t = cv2.solvePnP(obj_point.astype(float), img_point.astype(float), K, distCoeffs=None)
    assert retval == True
    R, Jacobian = cv2.Rodrigues(rvec)
    return R, t.flatten()


def estimate_cam_K_R_t_DLT(img_point, obj_point):
    init_cam_matrix, err = DLTcalib(3, obj_point, img_point)
    print("err", err)
    init_cam_matrix = init_cam_matrix.reshape((3, 4))

    A = init_cam_matrix[:3, :3]
    K, R = linalg.rq(A)
    t = linalg.solve(K, init_cam_matrix[:, 3])

    X = np.identity(3)
    for i in range(3):
        if K[i, i] < 0:
            K[:, i] = -K[:, i]
            X[i, i] = -1
    R = np.matmul(X, R)
    t = np.matmul(X, t)

    K /= K[2, 2]

    return K, R, t


if __name__ == "__main__":
    _test_convert_uvh_to_xyz()
