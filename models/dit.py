# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.model_util import RotaryEmbeddingND, get_2d_sincos_pos_embed, get_nd_sincos_pos_embed


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BottleneckPatchEmbed(nn.Module):
    """2D/3D input to patch embedding."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True, spatial_dims=2):
        super().__init__()
        img_size = (img_size,) * spatial_dims if isinstance(img_size, int) else tuple(img_size)
        patch_size = (patch_size,) * spatial_dims if isinstance(patch_size, int) else tuple(patch_size)
        assert len(img_size) == spatial_dims
        assert len(patch_size) == spatial_dims
        num_patches = math.prod(size // patch for size, patch in zip(img_size, patch_size))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.spatial_dims = spatial_dims

        conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.proj1 = conv(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = conv(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        spatial_shape = tuple(x.shape[2:])
        assert spatial_shape == self.img_size, \
            f"Input spatial size {spatial_shape} doesn't match model {self.img_size}."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        # NOTE: 1000.0 here is to scale the timestep to [0, 1000] as in DDPM
        args = (t[:, None] * 1000.0).float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2: # zero-padding the embedding by 1 when dim is odd.
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = nn.RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = rope(q)
        k = rope(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.,
            is_causal=False
        )
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, spatial_dims=2):
        super().__init__()
        patch_size = (patch_size,) * spatial_dims if isinstance(patch_size, int) else tuple(patch_size)
        self.norm_final = nn.RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, math.prod(patch_size) * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, condition):
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x,  condition, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiT(nn.Module):
    """
    Just image Transformer.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        out_channels=None,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=128,
        spatial_dims=2
    ):
        super().__init__()
        input_size = (input_size,) * spatial_dims if isinstance(input_size, int) else tuple(input_size)
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else int(out_channels)
        self.patch_size = (patch_size,) * spatial_dims if isinstance(patch_size, int) else tuple(patch_size)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.spatial_dims = spatial_dims

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)

        # linear embed
        self.x_embedder = BottleneckPatchEmbed(
            input_size, self.patch_size, in_channels, bottleneck_dim, hidden_size, bias=True, spatial_dims=spatial_dims
        )

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))

        # rope
        head_dim = hidden_size // num_heads
        axis_rope_dim = (head_dim // spatial_dims) // 2 * 2
        assert axis_rope_dim > 0, "Head dimension is too small for rotary embedding"
        patch_grid = tuple(size // patch for size, patch in zip(input_size, self.patch_size))
        self.feat_rope = RotaryEmbeddingND(
            dim=axis_rope_dim,
            grid_size=patch_grid,
            spatial_dims=spatial_dims,
            num_cls_token=0
        )

        # transformer
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # linear predict
        self.final_layer = FinalLayer(hidden_size, self.patch_size, self.out_channels, spatial_dims=spatial_dims)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        patch_grid = tuple(size // patch for size, patch in zip(self.input_size, self.patch_size))
        if self.spatial_dims == 2:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], patch_grid[0])
        else:
            pos_embed = get_nd_sincos_pos_embed(self.pos_embed.shape[-1], patch_grid, self.spatial_dims)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """
        x: (N, T, prod(patch_size) * C)
        output: (N, C, *spatial)
        """
        c = self.out_channels
        p = (p,) * self.spatial_dims if isinstance(p, int) else tuple(p)
        patch_grid = tuple(size // patch for size, patch in zip(self.input_size, p))
        assert math.prod(patch_grid) == x.shape[1]

        if self.spatial_dims == 2:
            h, w = patch_grid
            ph, pw = p
            x = x.reshape(shape=(x.shape[0], h, w, ph, pw, c))
            x = torch.einsum('nhwpqc->nchpwq', x)
            return x.reshape(shape=(x.shape[0], c, h * ph, w * pw))

        d, h, w = patch_grid
        pd, ph, pw = p
        x = x.reshape(shape=(x.shape[0], d, h, w, pd, ph, pw, c))
        x = torch.einsum('ndhwpqrc->ncdphqwr', x)
        return x.reshape(shape=(x.shape[0], c, d * pd, h * ph, w * pw))

    def forward(self, x, t):
        """
        x: (N, C, H, W) or (N, C, D, H, W)
        t: (N,)
        """
        # class and time embeddings
        t_emb = self.t_embedder(t)
        c = t_emb

        # forward DiT
        x = self.x_embedder(x)
        x += self.pos_embed

        for i, block in enumerate(self.blocks):
            x = block(x, c, self.feat_rope)

        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)

        return output

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, patch_size=4, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=512, patch_size=8, **kwargs)

def DiT_B_16(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, patch_size=16, **kwargs)

def DiT_B_32(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, patch_size=32, **kwargs)

def DiT_M_8(**kwargs):
    return DiT(depth=16, hidden_size=896, num_heads=14,
               bottleneck_dim=512, patch_size=8, **kwargs)

def DiT_L_16(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, patch_size=16, **kwargs)

def DiT_L_32(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, patch_size=32, **kwargs)

def DiT_H_16(**kwargs):
    return DiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, patch_size=16, **kwargs)

def DiT_H_32(**kwargs):
    return DiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, patch_size=32, **kwargs)


DiT_models = {
    'DiT-B/4': DiT_B_4,
    'DiT-B/8': DiT_B_8,
    'DiT-B/16': DiT_B_16,
    'DiT-B/32': DiT_B_32,
    'DiT-M/8': DiT_M_8,
    'DiT-L/16': DiT_L_16,
    'DiT-L/32': DiT_L_32,
    'DiT-H/16': DiT_H_16,
    'DiT-H/32': DiT_H_32,
}
