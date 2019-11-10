import logging
import os

import PIL.Image as pil
import numpy as np

from .mono_dataset import MonoDataset


class VideoFramesDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(VideoFramesDataset, self).__init__(*args, **kwargs)
        # K is the camera instrinsics matrix
        self.K = np.array([[0.78038, 0, 0.4912, 0],
                           [0, 1.38734, 0.5062, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        if self.camera_info_path:
            with np.load(self.camera_info_path) as data:
                self.K = data['K']

    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, panel_id):
        f_str = "frame_{:05d}_{}{}".format(frame_index, panel_id, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path
