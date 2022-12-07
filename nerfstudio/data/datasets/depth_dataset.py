# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Semantic dataset.
"""

from typing import Dict

import cv2
from numpy import float32

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Depths
from nerfstudio.data.datasets.base_dataset import *
from nerfstudio.data.utils.data_utils import get_1d_image_from_path


class DepthDataset(InputDataset):
    """Dataset that returns RGB and depth images.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs):
        super().__init__(dataparser_outputs)
        assert "depths" in dataparser_outputs.metadata.keys() and isinstance(
            dataparser_outputs.metadata["depths"], Depths
        )
        self.depths = dataparser_outputs.metadata["depths"]

    def get_image_depth(self, image_idx: int) -> TensorType["image_height", "image_width", "num_channels"]:
        """Returns a 3 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        filepath = self.depths.filenames[image_idx]
        scale = self.depths.scale
        depth = get_1d_image_from_path(filepath=filepath, scale=scale, dtype=float32)
        depth = depth.expand(-1, -1, 3)
        depth = depth - torch.min(depth)
        depth = depth / torch.max(depth)
        # depth_img = depth.cpu().detach().numpy()
        # depth_img = (depth_img * 255.0).astype(np.uint8)
        # cv2.imwrite("test.png", depth_img)
        # assert False

        return depth

    def get_metadata(self, data: Dict) -> Dict:
        # Load depth from depth image
        filepath = self.depths.filenames[data["image_idx"]]
        scale = self.depths.scale
        depth = get_1d_image_from_path(filepath=filepath, scale=scale, dtype=float32)

        return {"depth": depth}
