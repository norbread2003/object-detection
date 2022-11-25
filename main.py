#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ********************************************************************************
# © 2022 Yunlin Tan. All Rights Reserved.
# ********************************************************************************

"""
@package main

@brief Learning object detection segmentation with a DRL-based feature matching agent.

@author Yunlin Tan

@date 12/30/2021

**Related Page**: https://github.com/norbread2003/object-detection

Arguments
---------
    - None

Example Usage
-------------
    None

Update Record
-------------
0.1.0        12/30/2021   Yunlin Tan([None])            Object detection segmentation with a DRL-based matching agent.

Depends On
----------
**Python Dependencies:**
    - matplotlib
    - numpy

**Other Dependencies:**
    - None
"""
import time

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from Experiment import EConfig, Experiment
from utils import draw_obj_vertices

plt.rcParams.update({'font.size': 7})

matplotlib.use("webagg")

figure = {}
plot = {}
plot_handle = {}
center_color = {
    "C": "red",
    "D": "blue"
}
title = {
    "A": "Image A: picking scene",
    "A_depth": "depth of Image A",
    "B": "Image B: mesh renderer",
    "B_depth": "depth of Image B",
    "C": "Image C: resample from image A",
    "D": "Image D: resample from image B",
    "3d": "Centers of image C and D",
    "matched_point_info": "Matched Points Info",
    "E": "Image E: Object Pose Estimation"
}
if __name__ == "__main__":
    to_plot = ["A", "B", "C", "D", "3d", "matched_point_info", "E"]
    # to_plot = ["A", "A_depth", "B", "B_depth", "C", "D", "3d"]
    for i in to_plot:
        projection = None
        if i == "3d":
            projection = "3d"

        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_subplot(111, projection=projection)
        ax.set_title(title[i])
        figure[i] = fig
        plot[i] = ax
        if i == "3d":
            draw_obj_vertices(plot["3d"], "meshes/cocacola_can.obj")
            red_patch = mpatches.Patch(color='red', label='center of image C')
            blue_patch = mpatches.Patch(color='blue', label='center of image D')
            plot["3d"].legend(handles=[red_patch, blue_patch])

    ec = EConfig(
        "mujoco_models/world_0.xml",
        "mujoco_models/world_1.xml",
        "cam_0",
        "cam_1",
        image_C_D_size=100,
        visualize_joint=False
    )
    ex = Experiment(ec)

    for k in ["A", "B", "C", "D", "E"]:
        if k not in to_plot:
            continue
        plot_handle[k] = plot[k].imshow(ex.get_image(k))

    for k in ["A", "B"]:
        if k + "_depth" not in to_plot:
            continue
        plot_handle[k + "_depth"] = plot[k + "_depth"].imshow(np.array(ex.get_image(k, depth=True)))

    for k in ["C", "D"]:
        if k not in to_plot:
            continue
        plot_handle[k + "_center"] = plot["3d"].scatter(
            *ex.get_image_center_xyz_wrt_target_obj_frame(k),
            c=center_color[k])

    if "matched_point_info" in to_plot:
        plot["matched_point_info"].axis("off")
        plot_handle["matched_point_info"] = plot["matched_point_info"].text(0, 0,
                                                                            "[press m to declare a pair of matched points]")


    def press(event):
        start = time.time()
        k = event.key
        update = {
            "A": False, "B": False, "C": False, "D": False, "marked_point": False, "E": False
        }
        if k == "w":
            ex.move_image("C", "up")
            update["C"] = True
        elif k == "s":
            ex.move_image("C", "down")
            update["C"] = True
        elif k == "a":
            ex.move_image("C", "left")
            update["C"] = True
        elif k == "d":
            ex.move_image("C", "right")
            update["C"] = True
        elif k == "i":
            ex.move_image("D", "up")
            update["D"] = True
        elif k == "k":
            ex.move_image("D", "down")
            update["D"] = True
        elif k == "j":
            ex.move_image("D", "left")
            update["D"] = True
        elif k == "l":
            ex.move_image("D", "right")
            update["D"] = True
        elif k == "q":
            ex.zoom_image("C", "in")
            update["C"] = True
        elif k == "e":
            ex.zoom_image("C", "out")
            update["C"] = True
        elif k == "u":
            ex.zoom_image("D", "in")
            update["D"] = True
        elif k == "o":
            ex.zoom_image("D", "out")
            update["D"] = True
        elif k == "f":
            ex.rotate_mesh_in_image_B("left")
            update["B"] = True
        elif k == "h":
            ex.rotate_mesh_in_image_B("right")
            update["B"] = True
        elif k == "t":
            ex.rotate_mesh_in_image_B("up")
            update["B"] = True
        elif k == "g":
            ex.rotate_mesh_in_image_B("down")
            update["B"] = True
        elif k == "y":
            ex.rotate_mesh_in_image_B("clockwise")
            update["B"] = True
        elif k == "r":
            ex.rotate_mesh_in_image_B("anticlockwise")
            update["B"] = True
        elif k == "n":
            ex.step_simulation_in_image_A()
            update["A"] = True
        elif k == "m":
            ex.declare_match()
            update["C"] = True
            update["D"] = True
            update["marked_point"] = True
            update["E"] = True
        else:
            pass

        if update["E"] is True and "E" in to_plot:
            plot_handle["E"].set_data(ex.get_image("E"))
            figure["E"].canvas.draw_idle()

        if update["A"] is True or update["C"] is True:
            if update["A"] is True and "A" in to_plot:
                plot_handle["A"].set_data(ex.get_image("A"))
                figure["A"].canvas.draw_idle()
                if "A_depth" in to_plot:
                    plot_handle["A_depth"].set_data(np.array(ex.get_image("A", depth=True)))
                    figure["A_depth"].canvas.draw_idle()
            plot_handle["C"].set_data(ex.get_image("C"))
            plot_handle["C_center"].remove()
            plot_handle["C_center"] = plot["3d"].scatter(
                *ex.get_image_center_xyz_wrt_target_obj_frame("C"),
                c=center_color["C"])
            figure["C"].canvas.draw_idle()
            figure["3d"].canvas.draw_idle()

        if update["B"] is True or update["D"] is True:
            if update["B"] is True and "B" in to_plot:
                plot_handle["B"].set_data(ex.get_image("B"))
                figure["B"].canvas.draw_idle()
                if "B_depth" in to_plot:
                    plot_handle["B_depth"].set_data(np.array(ex.get_image("B", depth=True)))
                    figure["B_depth"].canvas.draw_idle()

            plot_handle["D"].set_data(ex.get_image("D"))
            plot_handle["D_center"].remove()
            plot_handle["D_center"] = plot["3d"].scatter(
                *ex.get_image_center_xyz_wrt_target_obj_frame("D"),
                c=center_color["D"])
            figure["D"].canvas.draw_idle()
            figure["3d"].canvas.draw_idle()

        if update["marked_point"] is True:
            uv, xyz, normal = ex.matched_point_info[-1]
            tmp = plot["3d"].get_xlim3d()
            scale = abs(tmp[1] - tmp[0]) / 5
            plot["3d"].quiver(*xyz, *(normal * scale))
            figure["3d"].canvas.draw_idle()

            v_1, v_2 = ex.matched_point_ground_truth[-1]
            idx = len(ex.matched_point_ground_truth) - 1
            dist = np.linalg.norm(v_1 - v_2, ord=2)
            new_s = "distance between matched pair {} is {}".format(str(idx), str(dist))
            s = plot_handle["matched_point_info"].get_text()
            plot_handle["matched_point_info"].set_text(s + "\n" + new_s)
            figure["matched_point_info"].canvas.draw_idle()

        print("Data update took ", time.time() - start)


    for _, fig in figure.items():
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        fig.canvas.mpl_connect('key_press_event', press)

    plt.show()
