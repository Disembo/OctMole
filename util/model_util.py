# --------------------------------------------------------
# References:
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------

import torch
from torch import nn
import numpy as np

from einops import rearrange, repeat


def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


class RotaryEmbeddingND(nn.Module):
    def __init__(
        self,
        dim,
        grid_size,
        spatial_dims=2,
        theta=10000,
        num_cls_token=0,
        **kwargs,
    ):
        super().__init__()
        if isinstance(grid_size, int):
            grid_size = (grid_size,) * spatial_dims
        assert len(grid_size) == spatial_dims
        assert dim % 2 == 0

        freqs_base = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        axis_freqs = []
        for size in grid_size:
            positions = torch.arange(size)
            freqs = torch.einsum('..., f -> ... f', positions.float(), freqs_base)
            freqs = repeat(freqs, '... n -> ... (n r)', r=2)
            axis_freqs.append(freqs)

        view_shapes = []
        for axis in range(spatial_dims):
            shape = [1] * (spatial_dims + 1)
            shape[axis] = grid_size[axis]
            shape[-1] = axis_freqs[axis].shape[-1]
            view_shapes.append(shape)

        freqs = broadcat(
            tuple(axis_freqs[axis].view(*view_shapes[axis]) for axis in range(spatial_dims)),
            dim=-1,
        )
        cos = freqs.cos().view(-1, freqs.shape[-1])
        sin = freqs.sin().view(-1, freqs.shape[-1])

        if num_cls_token > 0:
            cos_pad = torch.ones(num_cls_token, cos.shape[-1], dtype=cos.dtype)
            sin_pad = torch.zeros(num_cls_token, sin.shape[-1], dtype=sin.dtype)
            cos = torch.cat([cos_pad, cos], dim=0)
            sin = torch.cat([sin_pad, sin], dim=0)

        self.register_buffer("freqs_cos", cos)
        self.register_buffer("freqs_sin", sin)

    def forward(self, t):
        rot_dim = self.freqs_cos.shape[-1]
        t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
        t_rot = t_rot * self.freqs_cos + rotate_half(t_rot) * self.freqs_sin
        return torch.cat((t_rot, t_pass), dim=-1)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_nd_sincos_pos_embed(embed_dim, grid_size, spatial_dims=2):
    if isinstance(grid_size, int):
        grid_size = (grid_size,) * spatial_dims
    assert len(grid_size) == spatial_dims

    base_dim = (embed_dim // spatial_dims) // 2 * 2
    axis_dims = [base_dim] * spatial_dims
    axis_dims[-1] += embed_dim - sum(axis_dims)
    assert all(axis_dim % 2 == 0 for axis_dim in axis_dims), "Each positional axis dimension must be even"

    grids = np.meshgrid(*[np.arange(size, dtype=np.float32) for size in grid_size], indexing='ij')
    embeddings = [
        get_1d_sincos_pos_embed_from_grid(axis_dim, grid)
        for axis_dim, grid in zip(axis_dims, grids)
    ]
    return np.concatenate(embeddings, axis=1)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    # NOTE: change from 10000 to 100, as the data range is small
    omega = 1. / 100**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
