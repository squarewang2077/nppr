# Vision Transformers for small inputs
#
#   ViT    - "An Image is Worth 16x16 Words"      https://arxiv.org/abs/2010.11929
#   ConViT - "Improving Vision Transformers with
#             Soft Convolutional Inductive Biases" https://arxiv.org/abs/2103.10697
#
# Requirements:
#   torch >= 2.0
#
# Why these are written from scratch instead of wrapping torchvision:
#
#   * torchvision's VisionTransformer cannot express ConViT.  Its EncoderBlock
#     hard-codes `nn.MultiheadAttention`, so there is no way to substitute the
#     gated positional attention (GPSA) that ConViT is built on.
#   * torchvision's ViT has no stochastic depth — only `dropout` and
#     `attention_dropout`.  Stochastic depth is the main regulariser for
#     transformers trained from scratch on small datasets, which is exactly
#     what this project does.
#
# Since GPSA forces hand-written blocks anyway, ViT reuses the same ones: one
# code path, one `forward_features` contract, and drop_path for both.
#
# Sizing for this project's data: training runs at native resolution — 32px for
# CIFAR-10/100 and SVHN, 64px for TinyImageNet — so the patch size scales with
# the input to keep an 8x8 = 64 token grid throughout.  Stock 224px/patch-16
# models are both unusable at that size and far too heavy for adversarial
# training, which costs an extra K forward/backward passes per step.

import warnings

import torch
import torch.nn as nn

# ------------------------------------------------------------------
#                       Building Blocks
# ------------------------------------------------------------------


def default_patch_size(img_size: int) -> int:
    """Patch size giving an 8x8 = 64 token grid.

    32 (CIFAR-10/100, SVHN) -> 4;  64 (TinyImageNet) -> 8.

    Holding the token count fixed across datasets keeps the attention cost and
    the sequence length identical, so one training recipe transfers between
    them and only the patch projection changes size.
    """
    if img_size % 8 != 0:
        raise ValueError(f"img_size must be divisible by 8, got {img_size}")
    return img_size // 8


class DropPath(nn.Module):
    """Stochastic depth: drop the whole residual branch for a random subset of
    samples.  The main regulariser for transformers trained on small datasets.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        # One Bernoulli draw per sample, broadcast over tokens and channels.
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class Mlp(nn.Module):
    """Two-layer feed-forward block with GELU."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class PatchEmbed(nn.Module):
    """Split the image into non-overlapping patches via a strided conv.

    Returns (B, num_patches, embed_dim).
    """

    def __init__(self, img_size: int, patch_size: int, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} is not divisible by patch_size {patch_size}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        _, _, h, w = x.shape
        if h != self.img_size or w != self.img_size:
            raise ValueError(
                f"Input is {h}x{w} but this model was built for "
                f"{self.img_size}x{self.img_size}. Rebuild it with img_size={h}."
            )
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """Standard multi-head self-attention.

    Uses an explicit softmax rather than F.scaled_dot_product_attention: the
    fused kernels have no double-backward derivative, so this keeps
    grad-of-grad objectives available.  At 64 tokens the fused path would save
    very little anyway.
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                       # each (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    """Pre-norm transformer block.

    ``attn`` may be supplied to swap in a different attention module (ConViT
    passes a GPSA layer for its early blocks).
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0,
                 attn: nn.Module = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = attn if attn is not None else Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def init_vit_weights(module: nn.Module):
    """Truncated-normal init, as used by ViT/ConViT."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def drop_path_schedule(drop_path_rate: float, depth: int):
    """Linearly increasing stochastic-depth rate over the blocks."""
    return torch.linspace(0, drop_path_rate, depth).tolist()


def _warn_no_pretrained(name: str, pretrained: bool):
    """These are small-input models trained from scratch; no weights exist."""
    if pretrained:
        warnings.warn(
            f"{name}: no pretrained weights exist for small-input models; "
            "initialising from scratch.",
            stacklevel=3,
        )


# ------------------------------------------------------------------
#                       Vision Transformer
# ------------------------------------------------------------------


class VisionTransformer(nn.Module):
    """ViT with a class token and learned positional embeddings.

    Args:
        img_size:        input resolution (32 or 64 here).
        num_classes:     number of output classes.
        patch_size:      patch side; defaults to img_size // 8.
        embed_dim:       token width.
        depth:           number of transformer blocks.
        num_heads:       attention heads.
        mlp_ratio:       feed-forward expansion factor.
        drop_rate:       dropout on embeddings and inside the MLP.
        attn_drop_rate:  dropout on attention weights.
        drop_path_rate:  maximum stochastic-depth rate (linearly ramped).
    """

    def __init__(self, img_size: int = 32, num_classes: int = 10, patch_size: int = None,
                 in_chans: int = 3, embed_dim: int = 192, depth: int = 12, num_heads: int = 3,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0, drop_path_rate: float = 0.1):
        super().__init__()
        patch_size = patch_size or default_patch_size(img_size)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = drop_path_schedule(drop_path_rate, depth)
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(init_vit_weights)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled class-token feature, (B, embed_dim)."""
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        return self.norm(x)[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def vit_tiny(num_classes: int, pretrained: bool = False, img_size: int = 32,
             drop_path_rate: float = 0.1, **kwargs) -> VisionTransformer:
    """ViT-Tiny (192-dim, 12 layers, 3 heads), ~5.4M parameters.

    The lightest plain ViT that still behaves like one — the default choice for
    adversarial training, where cost scales with the number of attack steps.
    """
    _warn_no_pretrained("vit_tiny", pretrained)
    return VisionTransformer(img_size=img_size, num_classes=num_classes, embed_dim=192,
                             depth=12, num_heads=3, drop_path_rate=drop_path_rate, **kwargs)


def vit_small(num_classes: int, pretrained: bool = False, img_size: int = 32,
              drop_path_rate: float = 0.1, **kwargs) -> VisionTransformer:
    """ViT-Small (384-dim, 12 layers, 6 heads), ~21M parameters.

    Roughly ResNet-18/50 scale, and the standard ViT backbone in the
    adversarial-robustness literature.
    """
    _warn_no_pretrained("vit_small", pretrained)
    return VisionTransformer(img_size=img_size, num_classes=num_classes, embed_dim=384,
                             depth=12, num_heads=6, drop_path_rate=drop_path_rate, **kwargs)


# ------------------------------------------------------------------
#                              ConViT
# ------------------------------------------------------------------
#
# The early blocks use Gated Positional Self-Attention (GPSA) instead of plain
# attention.  GPSA mixes a content term (standard QK attention) with a purely
# positional term, under a learned per-head gate; the positional branch is
# initialised to mimic a convolution kernel.  The network therefore *starts*
# with a convolutional prior and can learn its way out of it, which is what
# makes ViTs trainable on datasets the size of CIFAR without huge pretraining.
#
# That property is the reason this model earns its place here: plain ViTs
# trained from scratch on 32px data are unstable, and adversarial training
# makes that worse.


class GPSA(nn.Module):
    """Gated Positional Self-Attention.

    attn = (1 - sigma(gate)) * softmax(QK) + sigma(gate) * softmax(pos)

    where `pos` is a learned projection of the relative offsets between every
    pair of patches.  `local_init` seeds that projection so each head starts
    as a distinct offset in a sqrt(num_heads) x sqrt(num_heads) convolution
    kernel, which is why num_heads must be a perfect square.

    Note this operates on patch tokens only — ConViT inserts the class token
    after the last GPSA block, since a class token has no spatial position.
    """

    def __init__(self, dim: int, num_heads: int, grid_size: int, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 locality_strength: float = 1.0):
        super().__init__()
        kernel_size = int(num_heads ** 0.5)
        if kernel_size ** 2 != num_heads:
            raise ValueError(
                f"GPSA needs a square number of heads (4, 9, 16, ...), got {num_heads}"
            )
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.locality_strength = locality_strength

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.pos_proj = nn.Linear(3, num_heads)
        self.gating_param = nn.Parameter(torch.ones(num_heads))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative offsets are fixed by the patch grid, so precompute once and
        # register as a buffer: it then follows .to(device) and .half().
        self.register_buffer("rel_indices", self._rel_indices(grid_size), persistent=False)

    @staticmethod
    def _rel_indices(grid_size: int) -> torch.Tensor:
        """(1, N, N, 3) tensor of (dx, dy, dx^2 + dy^2) for every patch pair."""
        n = grid_size ** 2
        ind = torch.arange(grid_size).view(1, -1) - torch.arange(grid_size).view(-1, 1)
        indx = ind.repeat(grid_size, grid_size)
        indy = ind.repeat_interleave(grid_size, dim=0).repeat_interleave(grid_size, dim=1)
        rel = torch.zeros(1, n, n, 3)
        rel[..., 0] = indx
        rel[..., 1] = indy
        rel[..., 2] = indx ** 2 + indy ** 2
        return rel

    def local_init(self):
        """Initialise the positional branch as a convolution kernel."""
        with torch.no_grad():
            self.v.weight.copy_(torch.eye(self.dim))
            kernel_size = int(self.num_heads ** 0.5)
            center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2
            for h1 in range(kernel_size):
                for h2 in range(kernel_size):
                    position = h1 + kernel_size * h2
                    self.pos_proj.weight[position, 0] = 2 * (h2 - center)
                    self.pos_proj.weight[position, 1] = 2 * (h1 - center)
                    self.pos_proj.weight[position, 2] = -1
            self.pos_proj.weight.mul_(self.locality_strength)

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk.unbind(0)

        content_score = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        pos_score = self.pos_proj(self.rel_indices).permute(0, 3, 1, 2).softmax(dim=-1)

        gate = torch.sigmoid(self.gating_param).view(1, -1, 1, 1)
        attn = (1.0 - gate) * content_score + gate * pos_score
        attn = attn / attn.sum(dim=-1, keepdim=True)
        return self.attn_drop(attn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class ConViT(nn.Module):
    """ViT whose first `local_up_to_layer` blocks use GPSA.

    Args:
        img_size:           input resolution (32 or 64 here).
        num_classes:        number of output classes.
        head_dim:           width *per head*; embed_dim = head_dim * num_heads.
        num_heads:          must be a perfect square (GPSA requirement).
        local_up_to_layer:  how many leading blocks use GPSA.
        locality_strength:  strength of the convolutional initialisation.
    """

    def __init__(self, img_size: int = 32, num_classes: int = 10, patch_size: int = None,
                 in_chans: int = 3, head_dim: int = 48, depth: int = 12, num_heads: int = 4,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0, drop_path_rate: float = 0.1,
                 local_up_to_layer: int = 10, locality_strength: float = 1.0):
        super().__init__()
        patch_size = patch_size or default_patch_size(img_size)
        embed_dim = head_dim * num_heads
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.local_up_to_layer = min(local_up_to_layer, depth)

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        grid_size = self.patch_embed.grid_size
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # No class-token slot: the class token joins after the GPSA blocks.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = drop_path_schedule(drop_path_rate, depth)
        blocks = []
        for i in range(depth):
            if i < self.local_up_to_layer:
                attn = GPSA(embed_dim, num_heads, grid_size, qkv_bias,
                            attn_drop_rate, drop_rate, locality_strength)
            else:
                attn = Attention(embed_dim, num_heads, qkv_bias, attn_drop_rate, drop_rate)
            blocks.append(Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[i], attn=attn))
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(init_vit_weights)
        # Must run after the generic init, which would otherwise overwrite it.
        for m in self.modules():
            if isinstance(m, GPSA):
                m.local_init()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled class-token feature, (B, embed_dim)."""
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        for i, blk in enumerate(self.blocks):
            if i == self.local_up_to_layer:
                x = torch.cat([cls_token, x], dim=1)
            x = blk(x)
        if self.local_up_to_layer >= len(self.blocks):
            # All blocks were GPSA, so the class token never joined.
            x = torch.cat([cls_token, x], dim=1)

        return self.norm(x)[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def convit_tiny(num_classes: int, pretrained: bool = False, img_size: int = 32,
                drop_path_rate: float = 0.1, **kwargs) -> ConViT:
    """ConViT-Tiny (4 heads x 48 = 192-dim, 12 layers), ~5.4M parameters."""
    _warn_no_pretrained("convit_tiny", pretrained)
    return ConViT(img_size=img_size, num_classes=num_classes, head_dim=48, num_heads=4,
                  depth=12, drop_path_rate=drop_path_rate, **kwargs)


def convit_small(num_classes: int, pretrained: bool = False, img_size: int = 32,
                 drop_path_rate: float = 0.1, **kwargs) -> ConViT:
    """ConViT-Small (9 heads x 48 = 432-dim, 12 layers), ~27M parameters."""
    _warn_no_pretrained("convit_small", pretrained)
    return ConViT(img_size=img_size, num_classes=num_classes, head_dim=48, num_heads=9,
                  depth=12, drop_path_rate=drop_path_rate, **kwargs)
