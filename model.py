"""
MCP-Net: Multidirectional Correlation-Propagating Network for Progressive
3D Medical Image Registration.

    H. Deng, J. Wang, X. Yin, X. Min.
    Pattern Recognition 183 (2027) 114973.
    https://doi.org/10.1016/j.patcog.2026.114973
"""

import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint


def _run(fn, *args, use_checkpoint=False):
    """Runs `fn(*args)`, with gradient checkpointing when enabled and training."""
    if use_checkpoint and args[0].requires_grad:
        return checkpoint.checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


########################################################
# Spatial transformation utilities
########################################################

class SpatialTransformer(nn.Module):
    """Warps `src` with a dense displacement field `flow` (the warping operator in Eq. (1))."""

    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grid = torch.stack(torch.meshgrid(vectors, indexing='ij')).unsqueeze(0).float()
        # Non-persistent: the identity grid is rebuilt from `size`, so it is not saved in checkpoints.
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # normalise grid values to [-1, 1] for grid_sample
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels to the last dim and reverse their order (grid_sample expects x, y, z)
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """Integrates a stationary velocity field by scaling and squaring (Sec. 3.5, Eqs. (5)-(6))."""

    def __init__(self, inshape, nsteps=7):
        super().__init__()
        assert nsteps >= 0, f'nsteps should be >= 0, found: {nsteps}'
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class FlowUpsample(nn.Module):
    """Resizes a displacement field and rescales its magnitude accordingly."""

    def __init__(self, factor=2, mode='trilinear'):
        super().__init__()
        self.factor = factor
        self.mode = mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x
        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
        return x


########################################################
# Shared-weight feature encoder (Sec. 3.1)
########################################################

class ConvBlock(nn.Module):
    """3x3x3 conv -> instance norm (optional) -> LeakyReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 alpha=0.1, norm=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels) if norm else nn.Identity()
        self.act = nn.LeakyReLU(alpha)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FeatureEncoder(nn.Module):
    """Five-level shared-weight conv encoder producing the feature pyramid {F^i}, i = 1, ..., 5.

    Output channels per level: 2c, 4c, 8c, 16c, 32c (c = `first_out_channels`).
    """

    def __init__(self, in_channels=1, first_out_channels=4):
        super().__init__()
        c = first_out_channels

        self.level1 = nn.Sequential(
            ConvBlock(in_channels, c, norm=False),
            ConvBlock(c, 2 * c),
            ConvBlock(2 * c, 2 * c),
        )
        self.level2 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvBlock(2 * c, 4 * c),
            ConvBlock(4 * c, 4 * c),
        )
        self.level3 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvBlock(4 * c, 8 * c),
            ConvBlock(8 * c, 8 * c),
        )
        self.level4 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvBlock(8 * c, 16 * c),
            ConvBlock(16 * c, 16 * c),
        )
        self.level5 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvBlock(16 * c, 32 * c),
            ConvBlock(32 * c, 32 * c),
        )

    def forward(self, x):
        f1 = self.level1(x)   # 1
        f2 = self.level2(f1)  # 1/2
        f3 = self.level3(f2)  # 1/4
        f4 = self.level4(f3)  # 1/8
        f5 = self.level5(f4)  # 1/16
        return f1, f2, f3, f4, f5


########################################################
# Multidirectional correlation (MC) estimator (Sec. 3.2, Fig. 3)
########################################################

class SubspaceSplitting(nn.Module):
    """Step i): linearly projects a feature map into K subspaces of size `head_dim`.

    Input (B, C, H, W, D) -> output (B, H, W, D, K * head_dim), channels-last.
    """

    def __init__(self, in_channels, dim):
        super().__init__()
        self.proj = nn.Linear(in_channels, dim)
        self.norm = nn.LayerNorm(dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat):
        feat = feat.permute(0, 2, 3, 4, 1)
        return self.norm(self.proj(feat))


class DirectionalPriorExtraction(nn.Module):
    """Step ii): multi-head directional prior extraction, Eq. (2).

    For every point P of the fixed subspace features F_{f,P}^k, a similarity weight
    S(F_{f,P}^k, F_{m,P}^k) is computed against the n x n x n neighbourhood of the moving
    subspace features, and used to weight the relative displacement vectors V_P of that
    neighbourhood. The result is one 3-D directional prior Dir^k per subspace (head).

    Adapted from the neighbourhood attention of ModeT (Wang et al., MICCAI 2023).

    Args:
        dim: total subspace dimension (K * head_dim).
        num_heads: number of subspaces K.
        kernel_size: neighbourhood size n.
    Returns:
        (B, 3K, H, W, D) tensor of concatenated directional priors.
    """

    def __init__(self, dim, num_heads, kernel_size=3, qk_scale=None, use_rpb=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.use_rpb = use_rpb
        if use_rpb:
            # relative position bias over the n x n x n neighbourhood
            self.rpb = nn.Parameter(torch.zeros(num_heads, kernel_size, kernel_size, kernel_size))

        # relative displacement vectors V_P of the neighbourhood
        vectors = [torch.arange(-s // 2 + 1, s // 2 + 1) for s in [kernel_size] * 3]
        grid = torch.stack(torch.meshgrid(vectors, indexing='ij'), -1).float()
        self.register_buffer('grid', grid, persistent=False)

    def _displacement_vectors(self, N):
        # (1, N, num_heads, n^3, 3)
        return (self.grid.reshape(self.kernel_size ** 3, 3)
                .unsqueeze(0).unsqueeze(0)
                .repeat(N, self.num_heads, 1, 1)
                .unsqueeze(0))

    def _add_position_bias(self, attn, N):
        # attn: (B, N, num_heads, 1, n^3)
        n3 = self.kernel_size ** 3
        bias_idx = torch.arange(n3).unsqueeze(-1).repeat(N, 1)
        bias = self.rpb.flatten(1, 3)[:, bias_idx].reshape(self.num_heads, N, 1, n3).transpose(0, 1)
        return attn + bias

    def forward(self, q, k):
        # q: fixed subspace features, k: moving subspace features, both (B, H, W, D, C)
        B, H, W, T, C = q.shape
        N = H * W * T
        num_tokens = int(self.kernel_size ** 3)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads, 1).transpose(3, 4) * self.scale  # B, N, heads, 1, head_dim
        pdr = (self.kernel_size - 1) // 2

        # gather the n x n x n neighbourhood of every point in the moving features
        k = k.permute(0, 4, 1, 2, 3)                                    # B, C, H, W, T
        k = nnf.pad(k, (pdr, pdr, pdr, pdr, pdr, pdr))                  # B, C, H+2, W+2, T+2
        k = k.flatten(0, 1)                                             # B*C, H+2, W+2, T+2
        k = (k.unfold(1, self.kernel_size, 1)
              .unfold(2, self.kernel_size, 1)
              .unfold(3, self.kernel_size, 1)
              .permute(0, 4, 5, 6, 1, 2, 3))                            # B*C, n, n, n, H, W, T
        k = k.reshape(B, self.num_heads, C // self.num_heads, num_tokens, N)
        k = k.permute(0, 4, 1, 3, 2)                                    # B, N, heads, n^3, head_dim

        # similarity S(F_f, F_m)
        attn = q @ k.transpose(-2, -1)                                  # B, N, heads, 1, n^3
        if self.use_rpb:
            attn = self._add_position_bias(attn, N)
        attn = attn.softmax(dim=-1)

        # weight the relative displacement vectors -> directional priors
        v = self._displacement_vectors(N)                               # 1, N, heads, n^3, 3
        x = attn @ v                                                    # B, N, heads, 1, 3
        x = x.reshape(B, H, W, T, self.num_heads * 3).permute(0, 4, 1, 2, 3)
        return x


class MLPBlock(nn.Module):
    """Linear -> GELU -> Dropout -> Linear -> Dropout."""

    def __init__(self, hidden_size, mlp_dim, dropout_rate=0.0):
        super().__init__()
        if not (0 <= dropout_rate <= 1):
            raise ValueError('dropout_rate should be between 0 and 1.')
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.drop1(self.act(self.linear1(x)))
        return self.drop2(self.linear2(x))


class DirectionEmbedding(nn.Module):
    """Step iii), first half: embeds [F_f, Dir^1..K, F_m] with Conv -> MLP -> channel attention,
    plus a shortcut connection (Fig. 3, "Direction embedding")."""

    def __init__(self, in_channels, out_channels, use_checkpoint=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.mlp = MLPBlock(hidden_size=out_channels, mlp_dim=out_channels * 2)
        self.channel_attention = RCAB(out_channels, use_checkpoint=use_checkpoint)

    def forward(self, x):
        x = self.conv(x)
        shortcut = x
        x = x.permute(0, 2, 3, 4, 1)  # B, H, W, D, C
        x = self.mlp(x)
        x = self.channel_attention(x)
        x = x.permute(0, 4, 1, 2, 3)  # B, C, H, W, D
        return x + shortcut


class MultiHeadCorrelationEstimator(nn.Module):
    """Step iii), second half: multi-head correlation estimator.

    A stack of (shifted-)window Transformer blocks whose attention heads correspond to the
    K directional priors. Outputs the correlation map C^j.
    Input / output: (B, C, D, H, W).
    """

    def __init__(self, embed_dim, num_layers, num_heads, window_size, mlp_ratio=4.0,
                 qkv_bias=True, drop=0.0, attn_drop=0.0, use_checkpoint=False):
        super().__init__()
        self.window_size = tuple(window_size)
        self.shift_size = tuple(i // 2 for i in window_size)
        no_shift = tuple(0 for _ in window_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim,
                             num_heads=num_heads,
                             window_size=self.window_size,
                             shift_size=no_shift if (i % 2 == 0) else self.shift_size,
                             mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias,
                             drop=drop,
                             attn_drop=attn_drop,
                             use_checkpoint=use_checkpoint)
            for i in range(num_layers)
        ])

    def forward(self, x_in):
        b, c, d, h, w = x_in.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        dp = int(math.ceil(d / window_size[0])) * window_size[0]
        hp = int(math.ceil(h / window_size[1])) * window_size[1]
        wp = int(math.ceil(w / window_size[2])) * window_size[2]
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x_in.device)

        x = einops.rearrange(x_in, 'b c d h w -> b d h w c')
        for block in self.blocks:
            x = block(x, mask_matrix=attn_mask)
        return einops.rearrange(x, 'b d h w c -> b c d h w')


class TransformerBlock(nn.Module):
    """(Shifted-)window multi-head self-attention block."""

    def __init__(self, embed_dim, num_heads, window_size, shift_size, mlp_ratio=4.0,
                 qkv_bias=True, drop=0.0, attn_drop=0.0, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WindowAttention(embed_dim,
                                    window_size=window_size,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    attn_drop=attn_drop,
                                    proj_drop=drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(hidden_size=embed_dim,
                            mlp_dim=int(embed_dim * mlp_ratio),
                            dropout_rate=drop)

    def _attention(self, x_in, mask_matrix):
        x = self.norm1(x_in)

        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = nnf.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)

        if any(i > 0 for i in shift_size):
            x_out = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x_out = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x_out = x_out[:, :d, :h, :w, :].contiguous()
        return x_out

    def _feed_forward(self, x_in):
        return self.mlp(self.norm2(x_in))

    def forward(self, x_in, mask_matrix=None):
        x = x_in + _run(self._attention, x_in, mask_matrix, use_checkpoint=self.use_checkpoint)
        x = x + _run(self._feed_forward, x, use_checkpoint=self.use_checkpoint)
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(self, embed_dim, num_heads, window_size, qkv_bias=False,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(torch.zeros(
            (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))

        coords = torch.stack(torch.meshgrid(torch.arange(window_size[0]),
                                            torch.arange(window_size[1]),
                                            torch.arange(window_size[2]), indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * window_size[2] - 1
        self.register_buffer('relative_position_index', relative_coords.sum(-1), persistent=False)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x_in, mask=None):
        b, n, c = x_in.shape
        qkv = self.qkv(x_in).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        attn = attn + relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn).to(v.dtype)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class MCEstimator(nn.Module):
    """Multidirectional correlation (MC) estimator, g(F_f^j, F_m^j) in Eq. (1).

    Args:
        feat_channels: channels of the encoder features at this level.
        num_subspaces: number of feature subspaces K.
        head_dim: dimension of each subspace.
        neighbourhood_size: local neighbourhood size n.
    Returns:
        Correlation map with 2 * feat_channels channels.
    """

    def __init__(self, feat_channels, num_subspaces, head_dim=6, neighbourhood_size=3,
                 window_size=(5, 5, 5), num_layers=4, qk_scale=None, use_checkpoint=False):
        super().__init__()
        dim = head_dim * num_subspaces
        corr_channels = 2 * feat_channels

        self.subspace_splitting = SubspaceSplitting(feat_channels, dim=dim)
        self.directional_prior = DirectionalPriorExtraction(dim, num_subspaces,
                                                            kernel_size=neighbourhood_size,
                                                            qk_scale=qk_scale)
        self.direction_embedding = DirectionEmbedding(2 * feat_channels + 3 * num_subspaces,
                                                      corr_channels,
                                                      use_checkpoint=use_checkpoint)
        self.correlation_estimator = MultiHeadCorrelationEstimator(embed_dim=corr_channels,
                                                                   num_layers=num_layers,
                                                                   num_heads=num_subspaces,
                                                                   window_size=window_size,
                                                                   use_checkpoint=use_checkpoint)

    def forward(self, feat_f, feat_m):
        # i) subspace splitting (shared projection for fixed and moving features)
        q = self.subspace_splitting(feat_f)
        k = self.subspace_splitting(feat_m)
        # ii) multi-head directional priors Dir^1..K
        dir_prior = self.directional_prior(q, k)
        # iii) correlation generation
        x = self.direction_embedding(torch.cat((feat_f, dir_prior, feat_m), dim=1))
        return self.correlation_estimator(x)


########################################################
# Correlation propagation (CP) module (Sec. 3.3, Fig. 4)
########################################################

class PatchExpanding(nn.Module):
    """Upsamples the prior correlation map C^{j+1} by 2 and halves its channels."""

    def __init__(self, embed_dim):
        super().__init__()
        self.up_conv = nn.ConvTranspose3d(embed_dim, embed_dim // 2, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(embed_dim // 2)

    def forward(self, x):
        x = self.up_conv(x)
        x = einops.rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b d h w c -> b c d h w')


class SpatialGatingUnit(nn.Module):
    """Spatial gating unit of gMLP. Input: (n, num_windows, window_volume, c)."""

    def __init__(self, c, n, use_bias=True):
        super().__init__()
        self.dense = nn.Linear(n, n, use_bias)
        self.norm = nn.LayerNorm(c // 2)

    def forward(self, x):
        u, v = torch.split(x, x.size(-1) // 2, dim=-1)
        v = self.norm(v)
        v = v.permute(0, 1, 3, 2)   # n, windows, c/2, window_volume
        v = self.dense(v)
        v = v.permute(0, 1, 3, 2)   # n, windows, window_volume, c/2
        return u * (v + 1.0)


class WindowGMLP(nn.Module):
    """gMLP applied within non-overlapping windows (Fig. 4, "In each window gMLP").
    Input / output: (n, h, w, d, c)."""

    def __init__(self, win_size, num_channels, factor=2, use_bias=True):
        super().__init__()
        self.fh, self.fw, self.fd = win_size

        self.norm = nn.LayerNorm(num_channels)
        self.in_project = nn.Linear(num_channels, num_channels * factor, use_bias)
        self.act = nn.GELU()
        self.sgu = SpatialGatingUnit(num_channels * factor, n=self.fh * self.fw * self.fd)
        self.out_project = nn.Linear(num_channels * factor // 2, num_channels, use_bias)

    def forward(self, x):
        _, h, w, d, _ = x.shape

        # pad so that each spatial dim is divisible by the window size
        pad_h = (self.fh - h % self.fh) % self.fh
        pad_w = (self.fw - w % self.fw) % self.fw
        pad_d = (self.fd - d % self.fd) % self.fd
        x = nnf.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))

        gh, gw, gd = x.shape[1] // self.fh, x.shape[2] // self.fw, x.shape[3] // self.fd
        x = split_images(x, patch_size=(self.fh, self.fw, self.fd))  # n, (gh gw gd), (fh fw fd), c

        shortcut = x
        x = self.norm(x)
        x = self.act(self.in_project(x))
        x = self.sgu(x)
        x = self.out_project(x)
        x = x + shortcut

        x = unsplit_images(x, grid_size=(gh, gw, gd), patch_size=(self.fh, self.fw, self.fd))
        if pad_h > 0 or pad_d > 0 or pad_w > 0:
            x = x[:, :h, :w, :d, :].contiguous()
        return x


class CompetitiveWeightModule(nn.Module):
    """Competitive weight module (CWM).

    Fuses the window-gMLP branches with weights produced by two 1x1x1 conv weight generators
    applied to the globally pooled sum of the branches, followed by a Softmax across branches.
    """

    def __init__(self, channels, num_branches=3, reduction=4):
        super().__init__()
        self.num_branches = num_branches
        self.conv1 = nn.Conv3d(channels, channels // reduction, kernel_size=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv3d(channels // reduction, channels * num_branches, kernel_size=1)

    def forward(self, branches):
        # branches: list of (n, h, w, d, c) tensors
        n, c = branches[0].shape[0], branches[0].shape[-1]
        s = sum(branches).permute(0, 4, 1, 2, 3).mean(dim=(2, 3, 4), keepdim=True)  # n, c, 1, 1, 1
        w = self.conv2(self.act(self.conv1(s)))                                     # n, c*B, 1, 1, 1
        w = w.reshape(n, c, self.num_branches).permute(2, 0, 1).softmax(dim=0)      # B, n, c
        w = w[:, :, None, None, None, :]                                            # B, n, 1, 1, 1, c
        return sum(b * w[i] for i, b in enumerate(branches))


class MultiWindowGMLP(nn.Module):
    """LayerNorm -> 3^3 / 5^3 / 7^3 window gMLPs -> CWM -> projection, with a residual.
    Input / output: (n, h, w, d, c)."""

    def __init__(self, num_channels, window_sizes=((3, 3, 3), (5, 5, 5), (7, 7, 7)),
                 use_bias=True, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = nn.LayerNorm(num_channels)
        self.window_gmlps = nn.ModuleList([
            WindowGMLP(win_size=ws, num_channels=num_channels, use_bias=use_bias) for ws in window_sizes
        ])
        self.cwm = CompetitiveWeightModule(num_channels, num_branches=len(window_sizes))
        self.out_project = nn.Linear(num_channels, num_channels, bias=use_bias)

    def _forward(self, x_in):
        x = self.norm(x_in)
        x = self.cwm([gmlp(x) for gmlp in self.window_gmlps])
        x = self.out_project(x)
        return x + x_in

    def forward(self, x_in):
        return _run(self._forward, x_in, use_checkpoint=self.use_checkpoint)


class SELayer(nn.Module):
    """Squeeze-and-excitation channel attention. Input / output: (n, h, w, d, c)."""

    def __init__(self, num_channels, reduction=4, use_bias=True):
        super().__init__()
        self.conv1 = nn.Conv3d(num_channels, num_channels // reduction, kernel_size=1, bias=use_bias)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(num_channels // reduction, num_channels, kernel_size=1, bias=use_bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_in):
        x = x_in.permute(0, 4, 1, 2, 3)
        x = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        w = self.sigmoid(self.conv2(self.relu(self.conv1(x))))
        return x_in * w.permute(0, 2, 3, 4, 1)


class RCAB(nn.Module):
    """Residual channel attention block: LN -> Conv -> LeakyReLU -> Conv -> SE, with a residual.
    Input / output: (n, h, w, d, c)."""

    def __init__(self, num_channels, reduction=4, lrelu_slope=0.2, use_bias=True, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = nn.LayerNorm(num_channels)
        self.conv1 = nn.Conv3d(num_channels, num_channels, kernel_size=3, bias=use_bias, padding='same')
        self.leaky_relu = nn.LeakyReLU(negative_slope=lrelu_slope)
        self.conv2 = nn.Conv3d(num_channels, num_channels, kernel_size=3, bias=use_bias, padding='same')
        self.se = SELayer(num_channels, reduction=reduction)

    def _forward(self, x):
        shortcut = x
        x = self.norm(x).permute(0, 4, 1, 2, 3)
        x = self.conv2(self.leaky_relu(self.conv1(x)))
        x = self.se(x.permute(0, 2, 3, 4, 1))
        return x + shortcut

    def forward(self, x):
        return _run(self._forward, x, use_checkpoint=self.use_checkpoint)


class CorrelationPropagation(nn.Module):
    """Correlation propagation (CP) module, H(., C^{j+1}) in Eq. (1).

    Concatenates the current-level MC output with the upsampled prior correlation map C^{j+1},
    then applies Conv -> multi-window gMLP (+ CWM) -> RCAB with a shortcut connection.
    """

    def __init__(self, mc_channels, prior_channels, out_channels, use_checkpoint=False):
        super().__init__()
        self.patch_expanding = PatchExpanding(prior_channels)
        self.conv = nn.Conv3d(mc_channels + prior_channels // 2, out_channels,
                              kernel_size=3, stride=1, padding='same')
        self.multi_window_gmlp = MultiWindowGMLP(out_channels, use_checkpoint=use_checkpoint)
        self.channel_attention = RCAB(out_channels, use_checkpoint=use_checkpoint)

    def forward(self, corr_mc, corr_prior):
        x = torch.cat((corr_mc, self.patch_expanding(corr_prior)), dim=1)
        x = self.conv(x)
        shortcut = x
        x = x.permute(0, 2, 3, 4, 1)  # n, h, w, d, c
        x = self.multi_window_gmlp(x)
        x = self.channel_attention(x)
        x = x.permute(0, 4, 1, 2, 3)  # n, c, h, w, d
        return x + shortcut


########################################################
# Registration head and finest-level correlation block
########################################################

class RegistrationHead(nn.Module):
    """Registration head R: predicts a 3-channel displacement (or velocity) field."""

    def __init__(self, in_channels, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv = nn.Conv3d(in_channels, 3, kernel_size=3, stride=1, padding='same')
        nn.init.normal_(self.conv.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return _run(self.conv, x, use_checkpoint=self.use_checkpoint)


class ConvCorrelationBlock(nn.Module):
    """Two Conv -> LeakyReLU -> InstanceNorm layers.

    Used at the finest level (j = 1) in place of the MC estimator, where neighbourhood
    attention at full resolution would be too memory-intensive.
    """

    def __init__(self, in_channels, out_channels, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act = nn.LeakyReLU(0.2)

    def _forward(self, x):
        x = self.norm1(self.act(self.conv1(x)))
        return self.norm2(self.act(self.conv2(x)))

    def forward(self, x):
        return _run(self._forward, x, use_checkpoint=self.use_checkpoint)


########################################################
# MCP-Net
########################################################

class MCPNet(nn.Module):
    """MCP-Net / MCP-Net-diff.

    Args:
        inshape: spatial size of the input volumes; each dim must be divisible by 16.
        in_channels: input image channels.
        channels: base channel width c of the encoder.
        num_subspaces: number of feature subspaces K at levels 5, 4, 3, 2 (coarse to fine).
        head_dim: dimension of each feature subspace.
        neighbourhood_size: local neighbourhood size n of the MC estimator.
        window_size: window size of the multi-head correlation estimator.
        num_transformer_layers: number of Transformer blocks in each correlation estimator.
        diffeomorphic: True -> MCP-Net-diff (the registration heads predict stationary velocity
            fields integrated by scaling and squaring); False -> MCP-Net (displacement fields).
        int_steps: number of scaling-and-squaring steps T.
        use_checkpoint: gradient checkpointing to reduce training memory.

    Forward:
        moving, fixed: (B, in_channels, *inshape)
        returns (warped moving image, final deformation field Phi_1 of shape (B, 3, *inshape))
    """

    def __init__(self,
                 inshape=(160, 192, 160),
                 in_channels=1,
                 channels=4,
                 num_subspaces=(8, 4, 2, 1),
                 head_dim=6,
                 neighbourhood_size=3,
                 window_size=(5, 5, 5),
                 num_transformer_layers=4,
                 diffeomorphic=True,
                 int_steps=7,
                 qk_scale=1,
                 use_checkpoint=True):
        super().__init__()
        assert len(num_subspaces) == 4, 'num_subspaces gives K for levels 5, 4, 3, 2'
        assert all(s % 16 == 0 for s in inshape), 'each dim of inshape must be divisible by 16'
        self.inshape = tuple(inshape)
        self.diffeomorphic = diffeomorphic
        c = channels

        self.encoder = FeatureEncoder(in_channels=in_channels, first_out_channels=c)

        self.mc_estimators = nn.ModuleDict()
        self.cp_modules = nn.ModuleDict()
        self.reg_heads = nn.ModuleDict()

        # Levels 5 (coarsest) to 2: MC estimator (+ CP module below level 5) + registration head
        for j, k in zip((5, 4, 3, 2), num_subspaces):
            feat_channels = c * 2 ** j          # encoder channels at level j
            corr_channels = 2 * feat_channels   # channels of the correlation map C^j
            assert corr_channels % k == 0, f'level {j}: {corr_channels} channels not divisible by K={k}'
            self.mc_estimators[f'level{j}'] = MCEstimator(feat_channels, k,
                                                          head_dim=head_dim,
                                                          neighbourhood_size=neighbourhood_size,
                                                          window_size=window_size,
                                                          num_layers=num_transformer_layers,
                                                          qk_scale=qk_scale,
                                                          use_checkpoint=use_checkpoint)
            if j < 5:
                self.cp_modules[f'level{j}'] = CorrelationPropagation(corr_channels, 2 * corr_channels,
                                                                      corr_channels, use_checkpoint)
            self.reg_heads[f'level{j}'] = RegistrationHead(corr_channels, use_checkpoint)

        # Level 1 (finest): conv correlation block instead of the MC estimator
        self.fine_correlation = ConvCorrelationBlock(4 * c, 4 * c, use_checkpoint)
        self.cp_modules['level1'] = CorrelationPropagation(4 * c, 8 * c, 2 * c, use_checkpoint)
        self.reg_heads['level1'] = RegistrationHead(2 * c, use_checkpoint)

        self.flow_upsample = FlowUpsample(factor=2, mode='trilinear')

        # Warping operators for levels 1-4 (index j-1)
        self.warpers = nn.ModuleList([SpatialTransformer(self._level_shape(j)) for j in range(1, 5)])
        # Scaling-and-squaring integrators for levels 1-5 (index j-1)
        if diffeomorphic:
            self.integrators = nn.ModuleList([VecInt(self._level_shape(j), nsteps=int_steps)
                                              for j in range(1, 6)])

    def _level_shape(self, j):
        return [s // 2 ** (j - 1) for s in self.inshape]

    def _predict_field(self, corr, j):
        """Registration head R at level j (+ integration for MCP-Net-diff)."""
        field = self.reg_heads[f'level{j}'](corr)
        if self.diffeomorphic:
            field = self.integrators[j - 1](field)
        return field

    def forward(self, moving, fixed):
        # Shared-weight feature encoder: feats[j-1] is the level-j feature
        feats_m = self.encoder(moving)
        feats_f = self.encoder(fixed)

        # Level 5 (coarsest): Phi_5 = R(g(F_f^5, F_m^5))
        corr = self.mc_estimators['level5'](feats_f[4], feats_m[4])
        phi = self._predict_field(corr, 5)
        phi = self.flow_upsample(phi)

        # Levels 4-2: Phi_j = R(H(g(F_f^j, F_m^j o Phi_{j+1}), C^{j+1}))
        for j in (4, 3, 2):
            warp = self.warpers[j - 1]
            feat_m = warp(feats_m[j - 1], phi)
            corr_mc = self.mc_estimators[f'level{j}'](feats_f[j - 1], feat_m)
            corr = self.cp_modules[f'level{j}'](corr_mc, corr)
            delta = self._predict_field(corr, j)
            phi = warp(phi, delta) + delta          # compose with the coarser deformation
            phi = self.flow_upsample(phi)

        # Level 1 (finest)
        warp = self.warpers[0]
        feat_m = warp(feats_m[0], phi)
        corr_mc = self.fine_correlation(torch.cat((feats_f[0], feat_m), dim=1))
        corr = self.cp_modules['level1'](corr_mc, corr)
        delta = self._predict_field(corr, 1)
        phi = warp(phi, delta) + delta

        moved = warp(moving, phi)
        return moved, phi


########################################################
# Helper functions
########################################################

def compute_mask(dims, window_size, shift_size, device):
    """Attention mask for shifted-window attention."""
    cnt = 0
    d, h, w = dims
    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


def window_partition(x_in, window_size):
    b, d, h, w, c = x_in.shape
    x = x_in.view(b,
                  d // window_size[0], window_size[0],
                  h // window_size[1], window_size[1],
                  w // window_size[2], window_size[2],
                  c)
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(
        -1, window_size[0] * window_size[1] * window_size[2], c)


def window_reverse(windows, window_size, dims):
    b, d, h, w = dims
    x = windows.view(b,
                     d // window_size[0], h // window_size[1], w // window_size[2],
                     window_size[0], window_size[1], window_size[2],
                     -1)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)


def get_window_size(x_size, window_size, shift_size=None):
    """Shrinks the window (and disables shifting) along dims smaller than the window."""
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    return tuple(use_window_size), tuple(use_shift_size)


def split_images(x, patch_size):
    """(n, h, w, d, c) -> (n, num_windows, window_volume, c)."""
    _, height, width, depth, _ = x.shape
    return einops.rearrange(
        x, 'n (gh fh) (gw fw) (gd fd) c -> n (gh gw gd) (fh fw fd) c',
        gh=height // patch_size[0], gw=width // patch_size[1], gd=depth // patch_size[2],
        fh=patch_size[0], fw=patch_size[1], fd=patch_size[2])


def unsplit_images(x, grid_size, patch_size):
    """(n, num_windows, window_volume, c) -> (n, h, w, d, c)."""
    return einops.rearrange(
        x, 'n (gh gw gd) (fh fw fd) c -> n (gh fh) (gw fw) (gd fd) c',
        gh=grid_size[0], gw=grid_size[1], gd=grid_size[2],
        fh=patch_size[0], fw=patch_size[1], fd=patch_size[2])


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    shape = (64, 64, 64)
    model = MCPNet(inshape=shape).to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M')
    moving = torch.randn(1, 1, *shape, device=device)
    fixed = torch.randn(1, 1, *shape, device=device)
    with torch.no_grad():
        moved, flow = model(moving, fixed)
    print('moved:', tuple(moved.shape), 'flow:', tuple(flow.shape))