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

"""Utility functions to allow easy re-use of common operations across dataloaders"""
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from rich.console import Console


def get_image_mask_tensor_from_path(filepath: Path, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Utility function to read a mask image from the given path and return a boolean tensor
    """
    pil_mask = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_mask.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_mask = pil_mask.resize(newsize, resample=Image.NEAREST)
    mask_tensor = torch.from_numpy(np.array(pil_mask)).unsqueeze(-1).bool()
    return mask_tensor


def get_semantics_and_mask_tensors_from_path(
    filepath: Path, mask_indices: Union[List, torch.Tensor], scale_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to read segmentation from the given filepath
    If no mask is required - use mask_indices = []
    """
    if isinstance(mask_indices, List):
        mask_indices = torch.tensor(mask_indices, dtype="int64").view(1, 1, -1)
    pil_image = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.NEAREST)
    semantics = torch.from_numpy(np.array(pil_image, dtype="int64"))[..., None]
    mask = torch.sum(semantics == mask_indices, dim=-1, keepdim=True) == 0
    return semantics, mask


def get_1d_image_from_path(filepath: Path, scale: float = 1.0, dtype: type = np.uint8) -> torch.Tensor:
    """
    Utility function to read any 1 dimensional input image like depth or infrared

    Args:
        filepath: path to image
        resolution: resolution of the 1 dimensional input
        type: datatype of the input image, [np.uint8, np.uint16, np.uint32] are supported
    """
    # assert dtype in [np.uint8, np.uint16, np.uint32, np.float32]
    # pil_image = Image.open(filepath)
    # image = np.array(pil_image, dtype=dtype)  # shape is (h, w)
    image = np.load(filepath)
    assert len(image.shape) == 2
    assert image.dtype == dtype
    # Reshape
    image = image[:, :, np.newaxis]
    assert len(image.shape) == 3
    # image = image[:, :, 0]
    assert image.shape[2] == 1
    # image = torch.from_numpy(image.astype("float32") * resolution)

    CONSOLE = Console(width=120)
    # CONSOLE.log(f"scale: {scale}")
    # max_val = np.max(image.astype("float32") * scale / float(np.iinfo(image.dtype).max))
    # min_val = np.min(image.astype("float32") * scale / float(np.iinfo(image.dtype).max))
    CONSOLE.log(f"max: {filepath}")
    CONSOLE.log(f"max: {np.min(image) * scale}")
    CONSOLE.log(f"max: {np.max(image) * scale}")
    CONSOLE.log(f"max: {image[0,0]}")
    image = torch.from_numpy(image.astype("float32") * scale)
    CONSOLE.log(f"max: {image[0,0,0]}")
    return image
