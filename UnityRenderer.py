import time

import PIL.Image
import matplotlib.pyplot as plt
from dm_control import mujoco

from mjremote import mjremote


class UnityRenderer(object):
    def __init__(self, physics, height=240, width=320, camera_id=-1):
        if type(camera_id) is str:
            camera_id = physics.model.name2id(camera_id, "camera")
        m = mjremote()
        m.connect()
        m.setcamera(camera_id, height, width)
        time.sleep(1)

        self.width = width
        self.height = height
        self.b = bytearray(3 * width * height)
        self.m = m
        self.physics = physics

    def set_qpos_and_render(self):
        self.m.setqpos(self.physics.data.qpos)
        self.m.getimage(self.b)
        im = PIL.Image.frombytes("RGB", (self.width, self.height), bytes(self.b)).transpose(PIL.Image.FLIP_TOP_BOTTOM)
        return im


if __name__ == "__main__":
    physics = mujoco.Physics.from_xml_path("mujoco_models/world_0.xml")
    ur = UnityRenderer(physics, camera_id="cam_0")
    im = ur.set_qpos_and_render()

    fig, ax = plt.subplots()
    handle = ax.imshow(im)


    def press(event):
        T = physics.data.time + 0.1
        while physics.data.time < T:
            physics.step()
        print(physics.data.time)
        im = ur.set_qpos_and_render()
        handle.set_data(im)
        fig.canvas.draw()


    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    fig.canvas.mpl_connect('key_press_event', press)
    plt.show()
    while True:
        plt.pause(10)
