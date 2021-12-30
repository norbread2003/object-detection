from enum import Enum

import numpy as np


class SamplingBoxAdjustment(Enum):
    NONE = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    ZOOM_IN = 5
    ZOOM_OUT = 6


class ResampleHelper(object):

    def __init__(self, im_size, resample_box_move_step_ratio=0.05, resample_box_zoom_step_ratio=0.05):
        self.resample_box_move_step_ratio = resample_box_move_step_ratio
        self.resample_box_zoom_step_ratio = resample_box_zoom_step_ratio
        self.w = im_size[0]
        self.h = im_size[1]
        self.resample_center = [self.w // 2, self.h // 2]
        self.resample_size = min(im_size) // 2
        self.resample_box = None

    def adjust_resmaple_box(self, box_adjustment: SamplingBoxAdjustment) -> tuple:
        move_step = int(self.resample_size * self.resample_box_move_step_ratio)
        zoom_step = int(self.resample_size * self.resample_box_zoom_step_ratio)
        if move_step == 0:
            move_step = 1
        if zoom_step == 0:
            zoom_step = 1

        if box_adjustment == SamplingBoxAdjustment.NONE:
            pass
        elif box_adjustment == SamplingBoxAdjustment.MOVE_UP:
            self.resample_center[1] -= move_step
        elif box_adjustment == SamplingBoxAdjustment.MOVE_DOWN:
            self.resample_center[1] += move_step
        elif box_adjustment == SamplingBoxAdjustment.MOVE_LEFT:
            self.resample_center[0] -= move_step
        elif box_adjustment == SamplingBoxAdjustment.MOVE_RIGHT:
            self.resample_center[0] += move_step
        elif box_adjustment == SamplingBoxAdjustment.ZOOM_IN:
            self.resample_size -= zoom_step
        elif box_adjustment == SamplingBoxAdjustment.ZOOM_OUT:
            self.resample_size += zoom_step
        else:
            raise Exception()

        if self.resample_size < 2:
            self.resample_size = 2

        if self.resample_size > min(self.w, self.h):
            self.resample_size = min(self.w, self.h)

        dist_to_boundaries = np.array((
            self.resample_center[0],
            self.w - self.resample_center[0],
            self.resample_center[1],
            self.h - self.resample_center[1],
        ))
        max_resample_size = dist_to_boundaries.min() * 2

        if self.resample_size > max_resample_size:
            if box_adjustment is not SamplingBoxAdjustment.ZOOM_OUT:
                # This is caused by MOVE_*, we will resolve the conflict by zooming
                self.resample_size = max_resample_size
            else:
                # This is caused by ZOOM_OUT, we will resolve the conflict by moving
                tmp = dist_to_boundaries - self.resample_size // 2
                if tmp[0] < 0:
                    self.resample_center[0] -= int(tmp[0])
                elif tmp[1] < 0:
                    self.resample_center[0] += int(tmp[1])

                if tmp[2] < 0:
                    self.resample_center[1] -= int(tmp[2])
                elif tmp[3] < 0:
                    self.resample_center[1] += int(tmp[3])

        upper_left_w = self.resample_center[0] - self.resample_size // 2
        upper_left_h = self.resample_center[1] - self.resample_size // 2

        bottom_right_w = self.resample_center[0] + self.resample_size // 2
        bottom_right_h = self.resample_center[1] + self.resample_size // 2

        self.resample_box = (upper_left_w, upper_left_h, bottom_right_w, bottom_right_h)

        return self.resample_box

    def get_resample_box(self) -> tuple:
        if self.resample_box is None:
            return self.adjust_resmaple_box(box_adjustment=SamplingBoxAdjustment.NONE)
        else:
            return self.resample_box
