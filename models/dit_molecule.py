import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dit import TimestepEmbedder, SwiGLUFFN, modulate


class CoordRoPE(nn.Module):
    def __init__(self, head_dim, axes=3, theta=10000.0):
        super().__init__()
        freq_dim = max((head_dim // 2) // axes, 1)
        self.axes = axes
        self.head_dim = head_dim
        self.register_buffer(
            "freqs",
            1.0 / (theta ** (torch.arange(freq_dim, dtype=torch.float32) / freq_dim)),
            persistent=False,
        )

    def forward(self, x, coords): 
        phases = torch.outer(coords.float().reshape(-1), self.freqs).reshape(*coords.shape[:-1], -1)
        need = self.head_dim // 2
        if phases.shape[-1] < need:
            phases = F.pad(phases, (0, need - phases.shape[-1]))
        phases = phases[..., :need]
        rot = torch.polar(torch.ones_like(phases), phases).unsqueeze(1)
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(xc * rot).reshape_as(x).to(x.dtype)


class MoleculeAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0.0, proj_drop=0.0, rope_axes=3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = attn_drop
        self.rope = CoordRoPE(self.head_dim, axes=rope_axes)

    def forward(self, x, coords, padding_mask=None):
        bsz, seqlen, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seqlen, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.rope(self.q_norm(q), coords)
        k = self.rope(self.k_norm(k), coords)
        
        attn_mask = None
        if padding_mask is not None:
            attn_mask = padding_mask[:, None, None, :].to(dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(attn_mask > 0, float("-inf"))
        
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        x = x.transpose(1, 2).reshape(bsz, seqlen, dim)
        
        return self.proj_drop(self.proj(x))


class MoleculeBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        rope_axes=3,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attn = MoleculeAttention(
            hidden_size, num_heads, attn_drop, proj_drop, rope_axes=rope_axes
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x, condition, coords, padding_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), coords, padding_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


def _insert_register_tokens(h, coords, padding_mask, register_tokens):
    """Prepend learned, non-spatial register tokens to a sparse sequence."""
    batch_size, _, hidden_size = h.shape
    num_registers = register_tokens.shape[1]
    registers = register_tokens.expand(batch_size, num_registers, hidden_size)
    register_coords = coords.new_zeros(batch_size, num_registers, coords.shape[-1])

    h = torch.cat([registers, h], dim=1)
    coords = torch.cat([register_coords, coords], dim=1)
    if padding_mask is not None:
        register_mask = padding_mask.new_zeros(batch_size, num_registers)
        padding_mask = torch.cat([register_mask, padding_mask], dim=1)
    return h, coords, padding_mask


class TaskHead(nn.Module):
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, out_dim, bias=False)
        )
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x):
        return self.net(x)


class MoleculeDiT(nn.Module):
    def __init__(
        self,
        type_vocab_size=118,
        coord_channels=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        rope_coord_scale=1.0,
        num_registers=8,
        register_start=4,
    ):
        super().__init__()
        self.type_vocab_size = int(type_vocab_size)
        self.rope_coord_scale = float(rope_coord_scale)
        self.num_registers = int(num_registers)
        self.register_start = int(register_start)
        if self.num_registers > 0:
            assert 0 <= self.register_start < depth
            self.register_tokens = nn.Parameter(
                torch.empty(1, self.num_registers, hidden_size)
            )
        else:
            self.register_tokens = None
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.type_embed = nn.Embedding(self.type_vocab_size, hidden_size)
        self.coord_embed = nn.Linear(coord_channels, hidden_size)
        self.blocks = nn.ModuleList([
            MoleculeBlock(hidden_size, num_heads, mlp_ratio, attn_drop, proj_drop)
            for _ in range(depth)
        ])
        self.norm = nn.RMSNorm(hidden_size)
        self.type_head = TaskHead(hidden_size, self.type_vocab_size)
        self.coord_head = TaskHead(hidden_size, coord_channels)
        self.initialize_weights()

    def initialize_weights(self):
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=0.02)
        nn.init.normal_(self.type_embed.weight, std=0.02)
        nn.init.normal_(self.coord_embed.weight, std=0.02)
        nn.init.zeros_(self.coord_embed.bias)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

    def forward(self, atom_type, coord, spvoxel, t, padding_mask=None):
        h = self.type_embed(atom_type) + self.coord_embed(coord)
        c = self.t_embedder(t)

        coords = spvoxel.float() / self.rope_coord_scale
        for block_idx, block in enumerate(self.blocks):
            if self.register_tokens is not None and block_idx == self.register_start:
                h, coords, padding_mask = _insert_register_tokens(
                    h, coords, padding_mask, self.register_tokens
                )
            h = block(h, c, coords, padding_mask)
        if self.register_tokens is not None:
            h = h[:, self.num_registers:]
        h = self.norm(h)
        
        return self.type_head(h), self.coord_head(h)


class MoleculeOctreeDiT(nn.Module):
    def __init__(
        self,
        max_depth,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        rope_coord_scale=2.0,
        self_condition=False,
        depth_increment=1,
        num_registers=8,
        register_start=4,
    ):
        super().__init__()
        self.rope_coord_scale = float(rope_coord_scale)
        self.self_condition = bool(self_condition)
        self.num_registers = int(num_registers)
        self.register_start = int(register_start)
        if self.num_registers > 0:
            assert 0 <= self.register_start < depth
            self.register_tokens = nn.Parameter(
                torch.empty(1, self.num_registers, hidden_size)
            )
        else:
            self.register_tokens = None
        self.out_channels = 8 ** int(depth_increment)
        self.input_embed = nn.Linear(
            self.out_channels * (2 if self.self_condition else 1), hidden_size
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.depth_embed = nn.Embedding(int(max_depth) + 1, hidden_size)
        self.blocks = nn.ModuleList([
            MoleculeBlock(hidden_size, num_heads, mlp_ratio, attn_drop, proj_drop)
            for _ in range(depth)
        ])
        self.norm = nn.RMSNorm(hidden_size)
        self.head = TaskHead(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=0.02)
        nn.init.normal_(self.input_embed.weight, std=0.02)
        nn.init.zeros_(self.input_embed.bias)
        nn.init.normal_(self.depth_embed.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

    def forward(self, x, spvoxel, t, depth, padding_mask=None, self_cond=None):
        if self.self_condition:
            if self_cond is None:
                self_cond = torch.zeros_like(x)
            x = torch.cat([x, self_cond], dim=-1)
        h = self.input_embed(x)
        c = self.t_embedder(t) + self.depth_embed(depth)
        coords = spvoxel.float() / self.rope_coord_scale
        for block_idx, block in enumerate(self.blocks):
            if self.register_tokens is not None and block_idx == self.register_start:
                h, coords, padding_mask = _insert_register_tokens(
                    h, coords, padding_mask, self.register_tokens
                )
            h = block(h, c, coords, padding_mask)
        if self.register_tokens is not None:
            h = h[:, self.num_registers:]
        return self.head(self.norm(h))


def MoleculeDiT_B(**kwargs):
    return MoleculeDiT(depth=12, hidden_size=768, num_heads=12, **kwargs)


def MoleculeDiT_S(**kwargs):
    return MoleculeDiT(depth=12, hidden_size=384, num_heads=6, **kwargs)


def MoleculeDiT_M(**kwargs):
    kwargs.setdefault("register_start", 5)
    return MoleculeDiT(depth=16, hidden_size=768, num_heads=12, **kwargs)


def MoleculeOctreeDiT_S(**kwargs):
    return MoleculeOctreeDiT(depth=12, hidden_size=384, num_heads=6, **kwargs)


def MoleculeOctreeDiT_B(**kwargs):
    return MoleculeOctreeDiT(depth=12, hidden_size=768, num_heads=12, **kwargs)


MoleculeDiT_models = {
    "MoleculeDiT-B": MoleculeDiT_B,
    "MoleculeDiT-S": MoleculeDiT_S,
    "MoleculeDiT-M": MoleculeDiT_M,
}


MoleculeOctreeDiT_models = {
    "MoleculeOctreeDiT-S": MoleculeOctreeDiT_S,
    "MoleculeOctreeDiT-B": MoleculeOctreeDiT_B,
}
