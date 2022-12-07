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
Collection of Losses.
"""

import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples

L1Loss = nn.L1Loss
MSELoss = nn.MSELoss

LOSSES = {"L1": L1Loss, "MSE": MSELoss}

EPS = 1.0e-7
SQRT_2 = torch.sqrt(torch.tensor(2))
SQRT_2PI = torch.sqrt(torch.tensor(2) * torch.pi)


def outer(
    t0_starts: TensorType[..., "num_samples_0"],
    t0_ends: TensorType[..., "num_samples_0"],
    t1_starts: TensorType[..., "num_samples_1"],
    t1_ends: TensorType[..., "num_samples_1"],
    y1: TensorType[..., "num_samples_1"],
) -> TensorType[..., "num_samples_0"]:
    """Faster version of

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L117
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L64

    Args:
        t0_starts: start of the interval edges
        t0_ends: end of the interval edges
        t1_starts: start of the interval edges
        t1_ends: end of the interval edges
        y1: weights
    """
    cy1 = torch.cat([torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1)

    idx_lo = torch.searchsorted(t1_starts.contiguous(), t0_starts.contiguous(), side="right") - 1
    idx_lo = torch.clamp(idx_lo, min=0, max=y1.shape[-1] - 1)
    idx_hi = torch.searchsorted(t1_ends.contiguous(), t0_ends.contiguous(), side="right")
    idx_hi = torch.clamp(idx_hi, min=0, max=y1.shape[-1] - 1)
    cy1_lo = torch.take_along_dim(cy1[..., :-1], idx_lo, dim=-1)
    cy1_hi = torch.take_along_dim(cy1[..., 1:], idx_hi, dim=-1)
    y0_outer = cy1_hi - cy1_lo

    return y0_outer


def lossfun_outer(
    t: TensorType[..., "num_samples+1"],
    w: TensorType[..., "num_samples"],
    t_env: TensorType[..., "num_samples+1"],
    w_env: TensorType[..., "num_samples"],
):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L136
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L80

    Args:
        t: interval edges
        w: weights
        t_env: interval edges of the upper bound enveloping historgram
        w_env: weights that should upper bound the inner (t,w) histogram
    """
    w_outer = outer(t[..., :-1], t[..., 1:], t_env[..., :-1], t_env[..., 1:], w_env)
    return torch.clip(w - w_outer, min=0) ** 2 / (w + EPS)


def ray_samples_to_sdist(ray_samples):
    """Convert ray samples to s space"""
    starts = ray_samples.spacing_starts
    ends = ray_samples.spacing_ends
    sdist = torch.cat([starts[..., 0], ends[..., -1:, 0]], dim=-1)  # (num_rays, num_samples + 1)
    return sdist


def interlevel_loss(weights_list, ray_samples_list):
    """Calculates the proposal loss in the MipNeRF-360 paper.

    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/model.py#L515
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/train_utils.py#L133
    """
    c = ray_samples_to_sdist(ray_samples_list[-1]).detach()
    w = weights_list[-1][..., 0].detach()
    loss_interlevel = 0.0
    for ray_samples, weights in zip(ray_samples_list[:-1], weights_list[:-1]):
        sdist = ray_samples_to_sdist(ray_samples)
        cp = sdist  # (num_rays, num_samples + 1)
        wp = weights[..., 0]  # (num_rays, num_samples)
        loss_interlevel += torch.mean(lossfun_outer(c, w, cp, wp))
    return loss_interlevel


# Verified
def lossfun_distortion(t, w):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L142
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L266
    """
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    return loss_inter + loss_intra


def distortion_loss(weights_list, ray_samples_list):
    """From mipnerf360"""
    c = ray_samples_to_sdist(ray_samples_list[-1])
    w = weights_list[-1][..., 0]
    loss = torch.mean(lossfun_distortion(c, w))
    return loss


def truncated_normal_distribution(
    x,
    sigma,
    a,
    b,
):
    """Truncated normal distribution"""
    standard_normal_distribution = lambda x: torch.exp(-(x**2) / 2) / SQRT_2PI
    cumulative_standard_normal_distribution = lambda x: (1 + torch.erf(x / SQRT_2)) / 2

    return standard_normal_distribution((x) / sigma) / (
        sigma
        * (cumulative_standard_normal_distribution((b) / sigma) - cumulative_standard_normal_distribution((a) / sigma))
    )


def cumulative_truncated_normal_distribution(
    x,
    sigma,
    a,
    b,
):
    """Truncated normal distribution"""
    standard_normal_distribution = lambda x: torch.exp(-(x**2) / 2) / SQRT_2PI
    cumulative_standard_normal_distribution = lambda x: (1 + torch.erf(x / SQRT_2)) / 2

    cumulative_distribution = (
        cumulative_standard_normal_distribution((x) / sigma) - cumulative_standard_normal_distribution(a / sigma)
    ) / ((cumulative_standard_normal_distribution((b) / sigma) - cumulative_standard_normal_distribution((a) / sigma)))
    heaviside_value = torch.tensor(1.0, device=x.device)
    truncated_distribution = (
        cumulative_distribution * torch.heaviside(x - a, heaviside_value) * torch.heaviside(b - x, heaviside_value)
    ) + torch.heaviside(x - b, torch.tensor(0.0, device=x.device))
    return truncated_distribution


def depth_loss(
    weights: TensorType[..., "num_samples", 1],
    ray_samples: RaySamples,
    termination_depth: TensorType[..., 1],
    far_plane: TensorType[..., 1],
    epsilon: TensorType[0],
) -> TensorType[..., 1]:
    """Composite samples along ray and calculate depths.

    Args:
        weights: Weights for each sample.
        ray_samples: Set of ray samples.
        ray_indices: Ray index for each sample, used when samples are packed.
        num_rays: Number of rays, used when samples are packed.

    Returns:
        Outputs of depth values.
    """
    # steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
    # lengths = (ray_samples.frustums.ends - ray_samples.frustums.starts) / 2
    # weights_sum = torch.reshape(1 - torch.sum(weights, dim=-2), (1, 1, 1))weights_sum.expand(4096, -1, -1)
    weights_sum = torch.sum(weights, dim=-2)
    # assert torch.min(weights_sum) > 0.9
    # weights_normalized = (weights) / (weights_sum[:, None] + 1e-10)
    # weights_far_plane = torch.clamp(1.0 - weights_sum, 1e-10, 1.0)
    # weights = torch.cat((weights, weights_sum.expand(4096, 1)[:, None]), -2)
    # depth = torch.cat((depth, far_plane), -2)
    # losses = -torch.log(weights) * torch.exp(
    #     -((depth - termination_depth[:, None]) ** 2) / (2)
    # )  #  * depth_error[:, None]
    # e = torch.tensor(0.1, device=termination_depth.device)
    # # epsilon = e
    sigma = epsilon / 3  # ** 2
    a = -epsilon
    b = epsilon
    # depths_weighted = torch.cumsum(steps, dim=-2)
    # weights_normalized / weights_sum[:, None]
    # loss = weights * depth_loss(depth, termination_depth[:, None]) / (weights_sum[:, None] + 1e-10)
    # depth = torch.sum(weights_normalized * steps, dim=-2)  # + (far_plane * (1.0 - weights_sum))
    # loss = torch.mean((depth - termination_depth) ** 2)

    # loss = (
    #     torch.sum((weights * (steps - termination_depth[:, None])) ** 2, dim=-2)
    #     + (((1.0 - weights_sum) * (far_plane - termination_depth)) ** 2)
    # ) / 49
    # mse_loss = MSELoss()
    # loss = mse_loss(weights_normalized * steps, weights_normalized * termination_depth[:, None])
    # loss = torch.mean((weights_normalized * (steps - termination_depth[:, None])) ** 2, dim=-2)
    # loss = torch.mean(loss)
    # normal = truncated_normal_distribution(torch.sub(depth, termination_depth[:, None]), sigma, a, b)
    # cumulative_normal = cumulative_truncated_normal_distribution(
    #     torch.sub(depth, termination_depth[:, None]), sigma, a, b
    # )
    # test = termination_depth[:, :, None].expand(-1, 48, -1)
    # depth_delta = torch.sub(depth, 10.0 * test)
    # losses_near = (
    #     weights - (truncated_normal_distribution(steps - termination_depth[:, None], sigma, a, b) * lengths)
    # ) ** 2
    # weights_cumulative = torch.cumsum(weights, -2)
    # weights_sum2 = weights_cumulative[..., -1, :]
    losses_cumulative = (
        weights
        - (
            (
                cumulative_truncated_normal_distribution(
                    ray_samples.frustums.ends - termination_depth[:, None], sigma, a, b
                )
                - cumulative_truncated_normal_distribution(
                    ray_samples.frustums.starts - termination_depth[:, None], sigma, a, b
                )
            )
        )
    ) ** 2
    # # test = cumulative_truncated_normal_distribution(
    # #     ray_samples.frustums.ends - termination_depth[:, None], sigma, a, b
    # # ) - cumulative_truncated_normal_distribution(ray_samples.frustums.starts - termination_depth[:, None], sigma, a, b)
    loss_cumulative = (torch.sum(losses_cumulative, dim=-2) + (1 - weights_sum) ** 2) / (
        losses_cumulative.size(dim=-2) + 1.0
    )
    loss_cumulative = torch.mean(loss_cumulative)
    # loss_near = (torch.sum(losses_near, dim=-2) + ((1 - weights_sum) ** 2)) / 49
    # loss_near = torch.mean(loss_near)

    # loss_weights = 1.0 - (weights_sum**2)
    # loss_weight = torch.mean(loss_weights)

    # m = nn.Sigmoid()
    # losses_empty = weights * m((10 * ((termination_depth[:, None] - depth) / e - 1)) ** 3)
    # loss_empty = torch.sum(losses_empty, dim=-2)
    # loss_empty = torch.mean(loss_empty)

    # # test = m((10 * ((depth - termination_depth[:, None]) / e - 1)) ** 3)
    # losses_dist = weights * m((10 * ((depth - termination_depth[:, None]) / e - 1)) ** 3)
    # loss_dist = torch.sum(losses_dist, dim=-2)
    # loss_dist = torch.mean(loss_dist)

    # test = torch.tensor(0.0, device=sigma.device)
    # for i in torch.arange(-2.0, 2.1, 0.1, device=sigma.device):
    #     test = torch.heaviside(i, torch.tensor(0.5, device=i.device))
    #     test2 = cumulative_truncated_normal_distribution(i, sigma, a, b)
    #     test2 *= 1
    # test = m((10 * i) ** 3)
    #     print(test)
    # test = truncated_normal_distribution(torch.tensor(-0.5), torch.tensor(0), sigma, torch.tensor(-1), torch.tensor(1))
    # test2 = truncated_normal_distribution(torch.tensor(0.5), torch.tensor(0), sigma, torch.tensor(-1), torch.tensor(1))
    # test3 = truncated_normal_distribution(torch.tensor(0), torch.tensor(0), sigma, torch.tensor(-1), torch.tensor(1))
    # test4 = truncated_normal_distribution(torch.tensor(2), torch.tensor(0), sigma, torch.tensor(-1), torch.tensor(1))

    # # (1 + torch.cos(x)) / 2 * torch.pi
    # last_depth = depth[..., -1, :]
    # losses = -(
    #     torch.log(weights + 1e-10)
    #     * torch.exp(-(((steps - termination_depth[:, None]) / 0.1) ** 2))
    #     * lengths  # / (2 * 1e-6)
    # )  # + 1e-10
    # loss = 10 * torch.mean((1.0 - weights_sum) ** 2)
    # far_plane_losses = (
    #     torch.log(1.0 - torch.clip(weights_sum, 1e-10, 1.0))
    #     * torch.exp(-(((1000.0 - termination_depth) / 1.0) ** 2))  # / (2 * 1e-6)
    #     * (1000.0 - steps[:, -1, :])
    # )  # + 1e-10 #  * depth_error[:, None]
    # test = torch.log(1.0 - weights_sum + 1e-10)
    # test = torch.log(1.0 - weights_sum + 1e-6)
    # test4 = 1.0 - weights_sum + 1e-6
    # test2 = torch.exp(-((100.0 - termination_depth) ** 2))
    # test3 = 100.0 - steps[:, -1, :]
    # losses += 1e-10
    # losses = torch.sum(losses, -2)
    # combined_losses = losses + far_plane_losses
    # loss = torch.mean(losses)  # + torch.mean((1.0 - weights_sum) ** 2)
    # assert not torch.any(torch.isnan(combined_losses))
    # return loss_near
    return loss_cumulative  # * (epsilon**2)  # + loss_near + loss_weight  #   + loss_empty + loss_dist


def nerfstudio_distortion_loss(
    ray_samples: RaySamples,
    densities: TensorType["bs":..., "num_samples", 1] = None,
    weights: TensorType["bs":..., "num_samples", 1] = None,
) -> TensorType["bs":..., 1]:
    """Ray based distortion loss proposed in MipNeRF-360. Returns distortion Loss.

    .. math::

        \\mathcal{L}(\\mathbf{s}, \\mathbf{w}) =\\iint\\limits_{-\\infty}^{\\,\\,\\,\\infty}
        \\mathbf{w}_\\mathbf{s}(u)\\mathbf{w}_\\mathbf{s}(v)|u - v|\\,d_{u}\\,d_{v}

    where :math:`\\mathbf{w}_\\mathbf{s}(u)=\\sum_i w_i \\mathbb{1}_{[\\mathbf{s}_i, \\mathbf{s}_{i+1})}(u)`
    is the weight at location :math:`u` between bin locations :math:`s_i` and :math:`s_{i+1}`.

    Args:
        ray_samples: Ray samples to compute loss over
        densities: Predicted sample densities
        weights: Predicted weights from densities and sample locations
    """
    if torch.is_tensor(densities):
        assert not torch.is_tensor(weights), "Cannot use both densities and weights"
        # Compute the weight at each sample location
        weights = ray_samples.get_weights(densities)
    if torch.is_tensor(weights):
        assert not torch.is_tensor(densities), "Cannot use both densities and weights"

    starts = ray_samples.spacing_starts
    ends = ray_samples.spacing_ends

    assert starts is not None and ends is not None, "Ray samples must have spacing starts and ends"
    midpoints = (starts + ends) / 2.0  # (..., num_samples, 1)

    loss = (
        weights * weights[..., None, :, 0] * torch.abs(midpoints - midpoints[..., None, :, 0])
    )  # (..., num_samples, num_samples)
    loss = torch.sum(loss, dim=(-1, -2))[..., None]  # (..., num_samples)
    loss = loss + 1 / 3.0 * torch.sum(weights**2 * (ends - starts), dim=-2)

    return loss
