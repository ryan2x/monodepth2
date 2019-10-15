import logging
import os

import PIL.Image as pil
import numpy as np

from .mono_dataset import MonoDataset


class VideoFramesDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(VideoFramesDataset, self).__init__(*args, **kwargs)
        # todo: what shall I put as K matrix?
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        f_str = "frame_{:05d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path
