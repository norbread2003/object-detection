import os
import textwrap

import numpy as np
from PIL import ImageDraw, Image, ImageFont
from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import enums
from scipy.spatial.transform import Rotation

from ResampleHelper import ResampleHelper
from ResampleHelper import SamplingBoxAdjustment
from utils import \
    draw_center_indicator, \
    quaternion_rotating_around_x_axis, \
    quaternion_rotating_around_y_axis, \
    quaternion_rotating_around_z_axis, \
    quaternion_multiply, \
    get_camera_intrinsic_matrix, \
    get_camera_frame_wrt_world, \
    uvh_to_X_cam, \
    X_cam_to_uv, \
    estimate_cam_K_R_t_opencvsolvePnP


class EConfig(object):
    def __init__(
            self,
            xml_1_path,
            xml_2_path,
            cam_1_name,
            cam_2_name,
            image_A_width=2000,
            image_A_height=2000,
            image_B_width=2000,
            image_B_height=2000,
            image_C_D_size=50,
            center_indicator_length=0.2,
            center_indicator_thick=0.03,
            resample_box_move_step_ratio=0.1,
            resample_box_zoom_step_ratio=0.1,
            mesh_rotation_step_in_degree=10,
            simulation_step_duration=0.1,
            matched_marker_ratio=0.15,
            visualize_joint=False
    ):
        assert os.path.isfile(xml_1_path)
        assert os.path.isfile(xml_2_path)
        self.xml_1_path = xml_1_path
        self.xml_2_path = xml_2_path

        assert type(cam_1_name) is str
        assert type(cam_2_name) is str
        self.cam_1_name = cam_1_name
        self.cam_2_name = cam_2_name

        assert type(image_A_width) is int
        assert type(image_A_height) is int
        assert type(image_B_width) is int
        assert type(image_B_height) is int
        self.image_A_width = image_A_width
        self.image_A_height = image_A_height
        self.image_B_width = image_B_width
        self.image_B_height = image_B_height

        assert type(image_C_D_size) is int
        self.image_C_D_size = image_C_D_size

        assert type(center_indicator_length) is float and 0 < center_indicator_length < 1
        assert type(center_indicator_thick) is float and 0 < center_indicator_thick < 1
        self.center_indicator_length = center_indicator_length
        self.center_indicator_thick = center_indicator_thick

        assert type(resample_box_move_step_ratio) is float and 0 < resample_box_move_step_ratio < 1
        assert type(resample_box_zoom_step_ratio) is float and 0 < resample_box_move_step_ratio < 1
        self.resample_box_move_step_ratio = resample_box_move_step_ratio
        self.resample_box_zoom_step_ratio = resample_box_zoom_step_ratio

        assert mesh_rotation_step_in_degree > 0
        self.mesh_rotation_step_in_degree = mesh_rotation_step_in_degree

        assert simulation_step_duration > 0
        self.simulation_step_duration = simulation_step_duration

        assert matched_marker_ratio > 0 and matched_marker_ratio < 1
        self.matched_marker_ratio = matched_marker_ratio

        assert type(visualize_joint) is bool
        self.visualize_joint = visualize_joint


class Experiment(object):
    image_adjustment = {
        "up": SamplingBoxAdjustment.MOVE_UP,
        "down": SamplingBoxAdjustment.MOVE_DOWN,
        "left": SamplingBoxAdjustment.MOVE_LEFT,
        "right": SamplingBoxAdjustment.MOVE_RIGHT,
        "in": SamplingBoxAdjustment.ZOOM_IN,
        "out": SamplingBoxAdjustment.ZOOM_OUT
    }

    def __init__(self, ec: EConfig):
        self.ec = ec
        self.physics = {
            "A": mujoco.Physics.from_xml_path(ec.xml_1_path),
            "B": mujoco.Physics.from_xml_path(ec.xml_2_path),
            "E": mujoco.Physics.from_xml_path(ec.xml_2_path),
        }

        self.cam_name = {
            "A": ec.cam_1_name,
            "B": ec.cam_2_name,
            "E": ec.cam_2_name,
        }

        self.image_wh = {
            "A": (ec.image_A_width, ec.image_A_height),
            "B": (ec.image_B_width, ec.image_B_height),
            "C": (ec.image_C_D_size, ec.image_C_D_size),
            "D": (ec.image_C_D_size, ec.image_C_D_size),
            "E": (ec.image_A_width, ec.image_A_height),
        }

        self.resample_helper = {
            "A": ResampleHelper(
                self.image_wh["A"],
                ec.resample_box_move_step_ratio,
                ec.resample_box_zoom_step_ratio),
            "B": ResampleHelper(
                self.image_wh["B"],
                ec.resample_box_move_step_ratio,
                ec.resample_box_zoom_step_ratio),
        }

        self.rgb_images = {}
        self.depth_pixels = {}
        self._stale = {
            "A": True, "B": True, "C": True, "D": True, "E": True
        }

        rotation_step = ec.mesh_rotation_step_in_degree
        self.mesh_rotation = {
            "up": quaternion_rotating_around_x_axis(rotation_step),
            "down": quaternion_rotating_around_x_axis(-rotation_step),
            "left": quaternion_rotating_around_y_axis(rotation_step),
            "right": quaternion_rotating_around_y_axis(-rotation_step),
            "clockwise": quaternion_rotating_around_z_axis(rotation_step),
            "anticlockwise": quaternion_rotating_around_z_axis(-rotation_step),
        }

        self.cam_K = {
            "A": get_camera_intrinsic_matrix(
                self.physics["A"],
                self.cam_name["A"],
                *self.image_wh["A"]),
            "B": get_camera_intrinsic_matrix(
                self.physics["B"],
                self.cam_name["B"],
                *self.image_wh["B"]),
        }

        self.cam_Rt = {
            "A": get_camera_frame_wrt_world(
                self.physics["A"], self.cam_name["A"]),
            "B": get_camera_frame_wrt_world(
                self.physics["B"], self.cam_name["B"]),
            "E": get_camera_frame_wrt_world(
                self.physics["E"], self.cam_name["E"]),

        }

        self.target_body_id = {
            "A": self.physics["A"].model.name2id("target", "body"),
            "B": self.physics["B"].model.name2id("target", "body"),
            "E": self.physics["E"].model.name2id("target", "body"),
        }

        self.matched_point_info = []
        self.matched_point_ground_truth = []

        rec_size = ec.image_C_D_size * ec.matched_marker_ratio
        self._marker_rec_size = rec_size
        self._marker_text_offset = np.array((rec_size * 0.2, 0))

        self._draw_rectangle_param = {
            "fill": "white",
            "outline": "black",
            "width": int(rec_size * 0.1)
        }

        self._draw_text_param = {
            "fill": "black",
            "font": ImageFont.truetype("Roboto-Bold.ttf", int(ec.image_C_D_size * ec.matched_marker_ratio)),
        }

        self.image_A_R_cam_estimate = None
        self.image_A_t_cam_estimate = None
        self.image_E_text = "please press m to collect matched points"

        scene_option = mujoco.wrapper.core.MjvOption()
        if ec.visualize_joint is True:
            scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True
            scene_option.flags[enums.mjtVisFlag.mjVIS_ACTUATOR] = True

        self.scene_option = scene_option

    def _draw_marker(self, im, loc: np.ndarray, text: str):
        d = ImageDraw.Draw(im)
        rec_upper_left = loc
        rec_bottom_right = loc + self._marker_rec_size
        if len(text) != 1:
            rec_bottom_right[0] += int(((len(text) - 1) * 0.5) * self._marker_rec_size)
        d.rectangle(
            (rec_upper_left[0], rec_upper_left[1], rec_bottom_right[0], rec_bottom_right[1]),
            **self._draw_rectangle_param)
        text_loc = loc + self._marker_text_offset
        d.text(text_loc, text, **self._draw_text_param)

    def _is_image_stale(self, image_name):
        if self._stale[image_name]:
            return True

        if image_name == "C":
            return self._is_image_stale("A")
        elif image_name == "D":
            return self._is_image_stale("B")
        else:
            return self._stale[image_name]

    def _clear_image_stale(self, image_name):
        assert image_name in ["A", "B", "C", "D", "E"]
        self._stale[image_name] = False

    def _set_image_stale(self, image_name):
        assert image_name in ["A", "B", "C", "D", "E"]
        self._stale[image_name] = True
        if image_name == "A":
            self._set_image_stale("C")
        elif image_name == "B":
            self._set_image_stale("D")

    def get_image(self, k, depth=False):
        if self._is_image_stale(k):
            if k == "E":
                if self.image_A_R_cam_estimate is None or self.image_A_t_cam_estimate is None:
                    im = Image.new("RGB", (200, 200), (0, 0, 0))
                    d = ImageDraw.Draw(im)
                    d.text((0, 0), text=textwrap.fill(self.image_E_text, 30), fill="white")
                    self.rgb_images[k] = im
                    self.depth_pixels[k] = im
                else:
                    R_obj_wrt_cam = self.image_A_R_cam_estimate
                    t_obj_wrt_cam = self.image_A_t_cam_estimate

                    R_cam_wrt_world, t_cam_wrt_world = self.cam_Rt["E"]

                    R_obj_wrt_world = R_cam_wrt_world.dot(R_obj_wrt_cam)
                    t_obj_wrt_world = R_cam_wrt_world.dot(t_obj_wrt_cam) + t_cam_wrt_world

                    q = Rotation.from_matrix(R_obj_wrt_world).as_quat()  # xyzw

                    with self.physics["E"].reset_context():
                        self.physics["E"].data.qpos[:3] = t_obj_wrt_world
                        self.physics["E"].data.qpos[3] = q[-1]
                        self.physics["E"].data.qpos[4:] = q[:3]

                    cam_name = self.cam_name[k]
                    self.rgb_images[k] = Image.fromarray(
                        self.physics[k].render(
                            height=self.image_wh[k][1],
                            width=self.image_wh[k][0],
                            camera_id=cam_name,
                            scene_option=self.scene_option
                        )
                    )
                    self.depth_pixels[k] = Image.fromarray(
                        self.physics[k].render(
                            height=self.image_wh[k][1],
                            width=self.image_wh[k][0],
                            camera_id=cam_name, depth=True)
                    )
                if depth:
                    ret = self.depth_pixels[k]
                else:
                    ret = self.rgb_images[k]
            elif k == "A" or k == "B":

                cam_name = self.cam_name[k]
                self.rgb_images[k] = Image.fromarray(
                    self.physics[k].render(
                        height=self.image_wh[k][1],
                        width=self.image_wh[k][0],
                        camera_id=cam_name,
                        scene_option=self.scene_option
                    )
                )
                self.depth_pixels[k] = Image.fromarray(
                    self.physics[k].render(
                        height=self.image_wh[k][1],
                        width=self.image_wh[k][0],
                        camera_id=cam_name, depth=True)
                )
                if depth:
                    ret = self.depth_pixels[k]
                else:
                    ret = self.rgb_images[k]
            elif k == "C" or "D":
                assert depth == False
                if k == "C":
                    src_image = "A"
                else:
                    src_image = "B"

                resample_helper = self.resample_helper[src_image]
                resample_box = resample_helper.get_resample_box()

                self.rgb_images[k] = draw_center_indicator(
                    self.get_image(src_image).resize(self.image_wh[k], box=resample_box),
                    self.ec.center_indicator_length,
                    self.ec.center_indicator_thick,
                )

                box_origin = np.array((resample_box[0], resample_box[1]))
                box_size = resample_box[2] - resample_box[0]
                img_size = self.rgb_images[k].size[0]

                if k == "C":
                    for idx, (uv, _, _) in enumerate(self.matched_point_info):
                        uv_new = ((uv - box_origin) * img_size / box_size).astype(int)
                        self._draw_marker(self.rgb_images[k], uv_new, str(idx))
                elif k == "D":
                    R_obj, t_obj = self._get_target_obj_frame_in_image("B")
                    R_cam, t_cam = self.cam_Rt["B"]
                    K = self.cam_K["B"]
                    for idx, (_, xyz, normal) in enumerate(self.matched_point_info):

                        # back face culling
                        normal_world = R_obj.dot(normal)
                        normal_cam = R_cam.transpose().dot(normal_world)

                        if normal_cam[2] > 0:
                            continue

                        X_world = R_obj.dot(xyz) + t_obj
                        X_cam = R_cam.transpose().dot((X_world - t_cam))

                        uv = X_cam_to_uv(X_cam, K)

                        uv_new = ((uv - box_origin) * img_size / box_size).astype(int)
                        self._draw_marker(self.rgb_images[k], uv_new, str(idx))
                else:
                    raise Exception()

                ret = self.rgb_images[k]
            else:
                raise Exception()
        else:
            if depth:
                assert k == "A" or k == "B"
                ret = self.depth_pixels[k]
            else:
                ret = self.rgb_images[k]
        self._clear_image_stale(k)
        return ret

    def _adjust_image(self, image_name, direction):
        if image_name == "C":
            src_image = "A"
        elif image_name == "D":
            src_image = "B"
        else:
            raise Exception()

        resample_helper = self.resample_helper[src_image]
        action = self.image_adjustment[direction]
        resample_helper.adjust_resmaple_box(action)

        self._set_image_stale(image_name)

    def move_image(self, image_name, direction):
        assert direction in ["up", "down", "left", "right"]
        self._adjust_image(image_name, direction)

    def zoom_image(self, image_name, direction):
        assert direction in ["in", "out"]
        self._adjust_image(image_name, direction)

    def rotate_mesh_in_image_B(self, direction):
        assert direction in ["up", "down", "left", "right", "clockwise", "anticlockwise"]

        q_1 = self.physics["B"].data.qpos[-4:]
        q_2 = self.mesh_rotation[direction]
        q_3 = quaternion_multiply(q_2, q_1)

        with self.physics["B"].reset_context():
            self.physics["B"].data.qpos[-4:] = q_3

        self._set_image_stale("B")

    def step_simulation_in_image_A(self):
        p = self.physics["A"]
        T = p.data.time + self.ec.simulation_step_duration
        while p.data.time < T:
            p.step()

        self._set_image_stale("A")

    def get_image_center_xyz_wrt_target_obj_frame(self, image_name):
        assert image_name == "C" or image_name == "D"
        if image_name == "C":
            src_image = "A"
        else:
            src_image = "B"

        resample_helper = self.resample_helper[src_image]
        w, h = resample_helper.resample_center
        d = self.get_image(src_image, depth=True).getpixel((w, h))

        K = self.cam_K[src_image]
        R_cam, t_cam = self.cam_Rt[src_image]

        X_cam = uvh_to_X_cam(w, h, d, K)
        X_world = np.matmul(R_cam, X_cam) + t_cam

        R_obj, t_obj = self._get_target_obj_frame_in_image(src_image)
        X_obj = np.matmul(R_obj.transpose(), X_world - t_obj)

        return X_obj

    def get_image_D_center_normal_wrt_target_obj_frame(self):
        src_image = "B"
        K = self.cam_K[src_image]
        R_cam, t_cam = self.cam_Rt[src_image]
        resample_helper = self.resample_helper[src_image]
        w, h = resample_helper.resample_center
        depth_img = self.get_image(src_image, depth=True)
        d = depth_img.getpixel((w, h))
        p_1 = uvh_to_X_cam(w, h, d, K)

        candiates = ((w, h - 1), (w - 1, h), (w, h + 1), (w + 1, h))
        for idx in range(4):
            i_2 = idx
            i_3 = (i_2 + 1) % 4
            uv_2 = candiates[i_2]
            uv_3 = candiates[i_3]
            # TODO: if p_2 or p_3 is not on target object, continue
            d_2 = depth_img.getpixel(uv_2)
            d_3 = depth_img.getpixel(uv_3)

            p_2 = uvh_to_X_cam(*uv_2, d_2, K)
            p_3 = uvh_to_X_cam(*uv_3, d_3, K)
            break

        v_1 = p_2 - p_1
        v_2 = p_3 - p_1

        normal_cam = np.cross(v_1, v_2)  # in camera frame

        normal_world = R_cam.dot(normal_cam)

        R_obj, _ = self._get_target_obj_frame_in_image("B")

        normal_obj = R_obj.transpose().dot(normal_world)

        normal_obj /= np.linalg.norm(normal_obj)

        return normal_obj

    def _get_target_obj_frame_in_image(self, image_name):
        assert image_name in ["A", "B", "E"]
        data = self.physics[image_name].data
        bid = self.target_body_id[image_name]

        t = data.xpos[bid]
        R = data.xmat[bid].reshape((3, 3))

        return R, t

    def declare_match(self):
        resample_helper = self.resample_helper["A"]
        w, h = resample_helper.resample_center

        uv = np.array((w, h))
        xyz_D = self.get_image_center_xyz_wrt_target_obj_frame("D")

        normal = self.get_image_D_center_normal_wrt_target_obj_frame()

        self.matched_point_info.append((uv, xyz_D, normal))

        self.matched_point_ground_truth.append(
            (self.get_image_center_xyz_wrt_target_obj_frame("C"), xyz_D)
        )

        try:
            img_point = np.array([i[0] for i in self.matched_point_info])
            obj_point = np.array([i[1] for i in self.matched_point_info])

            np.save("img_point", img_point)
            np.save("obj_point", obj_point)
            # img_point = np.load("img_point.npy")
            # obj_point = np.load("obj_point.npy")

            K_true = self.cam_K["A"]
            R, t = estimate_cam_K_R_t_opencvsolvePnP(img_point, obj_point, K_true)
            self.image_A_R_cam_estimate = R
            self.image_A_t_cam_estimate = t


        except Exception as e:
            s = "Estimation error:" + str(e)
            print(s)
            self.image_E_text = s
            self.image_A_R_cam_estimate = None
            self.image_A_t_cam_estimate = None

        print("image_A_R_cam_estimate=", self.image_A_R_cam_estimate)
        print("image_A_t_cam_estimate=", self.image_A_t_cam_estimate)

        self._set_image_stale("C")
        self._set_image_stale("D")
        self._set_image_stale("E")


if __name__ == "__main__":
    ec = EConfig("mujoco_models/world_0.xml", "mujoco_models/world_1.xml", "cam_0", "cam_1")
    ex = Experiment(ec)
    # ex.get_image("C").show()
    # ex.get_image("D").show()
    # ex.get_image("A").show()
    # ex.get_image("B").show()
    # ex.move_image("C", "up")
    # ex.move_image("D", "up")
    # ex.get_image("C").show()
    # ex.get_image("D").show()

    # ex.get_image("D").show()
    # ex.get_image("B").show()
    # ex.rotate_mesh_in_image_B("left")
    # ex.get_image("D").show()
    # ex.get_image("B").show()
    # ex.rotate_mesh_in_image_B("up")
    # ex.get_image("D").show()
    # ex.get_image("B").show()

    # ex.get_image("C").show()
    # ex.get_image("A").show()
    # ex.step_simulation_in_image_A()
    # ex.get_image("C").show()
    # ex.get_image("A").show()
    # ex.step_simulation_in_image_A()
    # ex.get_image("C").show()
    # ex.get_image("A").show()

    print(ex.get_image_center_xyz_wrt_target_obj_frame("C"))
    print(ex.get_image_center_xyz_wrt_target_obj_frame("D"))
