"""
Time-only Heavy LinOSS AframeModel wrapping a local Backbone with pooling
in a PyTorch Lightning module, configurable via jsonargparse.
"""

import logging
import math

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import nn, random
from jax.nn.initializers import normal
import jax

from jaxtyping import Array, PRNGKeyArray


logger = logging.getLogger(__name__)


class _Downsample1D(eqx.Module):
    conv: eqx.nn.Conv1d
    norm: eqx.nn.GroupNorm

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        norm_groups: int,
        *,
        key: PRNGKeyArray,
    ):
        self.conv = eqx.nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=1,
            stride=stride,
            use_bias=False,
            key=key,
        )
        self.norm = eqx.nn.GroupNorm(norm_groups, out_ch)

    def __call__(self, x: Array) -> Array:
        return self.norm(self.conv(x))


class BasicBlock1D(eqx.Module):
    """Residual basic block with two 1D convolutions and GroupNorm."""

    conv1: eqx.nn.Conv1d
    norm1: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv1d
    norm2: eqx.nn.GroupNorm
    downsample: _Downsample1D | None

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm_groups: int = 16,
        downsample: "_Downsample1D | None" = None,
        *,
        key: PRNGKeyArray,
    ):
        k1, k2 = jr.split(key)
        padding = kernel_size // 2
        self.conv1 = eqx.nn.Conv1d(
            inplanes,
            planes,
            kernel_size,
            stride=stride,
            padding=padding,
            use_bias=False,
            key=k1,
        )
        self.norm1 = eqx.nn.GroupNorm(norm_groups, planes)
        self.conv2 = eqx.nn.Conv1d(
            planes,
            planes,
            kernel_size,
            padding=padding,
            use_bias=False,
            key=k2,
        )
        self.norm2 = eqx.nn.GroupNorm(norm_groups, planes)
        self.downsample = downsample

    def __call__(self, x: Array) -> Array:
        identity = x
        out = jax.nn.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return jax.nn.relu(out + identity)


def _make_layer_1d(
    inplanes: int,
    planes: int,
    num_blocks: int,
    kernel_size: int,
    stride: int = 1,
    norm_groups: int = 16,
    *,
    key: PRNGKeyArray,
) -> tuple[list[BasicBlock1D], int]:
    # key[0] → downsample, key[1..] → blocks
    keys = jr.split(key, num_blocks + 1)

    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = _Downsample1D(
            inplanes, planes, stride, norm_groups, key=keys[0]
        )

    blocks = [
        BasicBlock1D(
            inplanes,
            planes,
            kernel_size,
            stride=stride,
            norm_groups=norm_groups,
            downsample=downsample,
            key=keys[1],
        )
    ]
    for i in range(1, num_blocks):
        blocks.append(
            BasicBlock1D(
                planes,
                planes,
                kernel_size,
                norm_groups=norm_groups,
                key=keys[i + 1],
            )
        )

    return blocks, planes


class ResNet1D(eqx.Module):
    """1D ResNet (BasicBlock) matching ml4gw ResNet1D with GroupNorm.

    Input:  (num_ifos, time)
    Output: (classes,)
    """

    conv1: eqx.nn.Conv1d
    norm1: eqx.nn.GroupNorm
    maxpool: eqx.nn.MaxPool1d
    res_layers: list  # list[list[BasicBlock1D]]
    avgpool: eqx.nn.AdaptiveAvgPool1d
    fc: eqx.nn.Linear

    def __init__(
        self,
        in_channels: int,
        layers: tuple[int, ...],
        classes: int,
        kernel_size: int = 3,
        norm_groups: int = 16,
        *,
        key: PRNGKeyArray,
    ):
        k_stem, k_layers, k_fc = jr.split(key, 3)
        layer_keys = jr.split(k_layers, len(layers))

        self.conv1 = eqx.nn.Conv1d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            use_bias=False,
            key=k_stem,
        )
        self.norm1 = eqx.nn.GroupNorm(norm_groups, 64)
        self.maxpool = eqx.nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        inplanes = 64
        res_layers = []
        for i, (num_blocks, k) in enumerate(
            zip(layers, layer_keys, strict=True)
        ):
            planes = 64 * (2**i)
            stride = 1 if i == 0 else 2
            blocks, inplanes = _make_layer_1d(
                inplanes,
                planes,
                num_blocks,
                kernel_size,
                stride,
                norm_groups,
                key=k,
            )
            res_layers.append(blocks)

        self.res_layers = res_layers
        self.avgpool = eqx.nn.AdaptiveAvgPool1d(target_shape=1)
        self.fc = eqx.nn.Linear(inplanes, classes, key=k_fc)

    def __call__(self, x: Array) -> Array:
        # x: (num_ifos, time)
        x = jax.nn.relu(self.norm1(self.conv1(x)))
        x = self.maxpool(x)
        for layer in self.res_layers:
            for block in layer:
                x = block(x)
        x = self.avgpool(x)  # (channels, 1)
        x = x[..., 0]  # (channels,)
        return self.fc(x)  # (classes,)


class GLU(eqx.Module):
    """Gated Linear Unit."""

    w1: eqx.nn.Linear
    w2: eqx.nn.Linear

    def __init__(self, input_dim: int, output_dim: int, *, key: PRNGKeyArray):
        k1, k2 = jr.split(key)
        self.w1 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=k1)
        self.w2 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=k2)

    def __call__(self, x: Array) -> Array:
        return self.w1(x) * jax.nn.sigmoid(self.w2(x))


@jax.vmap
def _binary_op(q_i, q_j):
    """Associative operator for the 2nd-order state recurrence."""
    A_i, b_i = q_i
    A_j, b_j = q_j
    N = A_i.size // 4
    iA, iB, iC, iD = A_i[:N], A_i[N : 2 * N], A_i[2 * N : 3 * N], A_i[3 * N :]
    jA, jB, jC, jD = A_j[:N], A_j[N : 2 * N], A_j[2 * N : 3 * N], A_j[3 * N :]
    A_new = jnp.concatenate(
        [
            jA * iA + jB * iC,
            jA * iB + jB * iD,
            jC * iA + jD * iC,
            jC * iB + jD * iD,
        ]
    )
    b1, b2 = b_i[:N], b_i[N:]
    b_new = jnp.concatenate([jA * b1 + jB * b2, jC * b1 + jD * b2])
    return A_new, b_new + b_j


def _damped_linoss_imex(A_diag, G_diag, B, x, step):
    """Damped LinOSS-IMEX via parallel scan.

    Args:
        A_diag: (P,)   diagonal stiffness
        G_diag: (P,)   diagonal damping
        B:      (P, H) complex input matrix
        x:      (T, H) real input sequence
        step:   (P,)   discretisation step-sizes

    Returns:
        ys: (T, P) complex — velocity component of the state
    """
    P = A_diag.shape[0]
    L = x.shape[0]
    Bu = jax.vmap(lambda u: B @ u)(x)  # (T, P) complex
    S = 1.0 + step * G_diag
    M = jnp.concatenate(
        [1 / S, -step * A_diag / S, step / S, 1.0 - step**2 * A_diag / S]
    )
    M_elems = M * jnp.ones((L, 4 * P))
    F = jnp.hstack([Bu * step / S, Bu * step**2 / S])
    _, xs = jax.lax.associative_scan(_binary_op, (M_elems, F))
    return xs[:, P:]  # velocity component


def _theta_to_A(theta, G, steps):
    """Map oscillation angle → initial A_diag for damped LinOSS-IMEX."""
    cos2 = jnp.cos(theta) ** (-2)
    sq = jnp.sqrt(steps**4 * cos2 + steps**5 * G * cos2)
    tan2 = jnp.tan(theta) ** 2
    base = -(steps**2) * (-4 - 2 * steps * G - (4 + 2 * steps * G) * tan2)
    denom = 2 * steps**4 * (1 + tan2)
    A_plus = (4 * sq + base) / denom
    A_minus = (-4 * sq + base) / denom
    return jnp.where(theta > jnp.pi / 2, A_plus, A_minus)


class LinOSSMixer(eqx.Module):
    """LinOSS-IMEX with damping (matches base.yaml sequence_mixers config)."""

    A_diag: Array  # (P,)
    G_diag: Array  # (P,)
    B: Array  # (P, H, 2) — complex stored as real/imag pair
    C: Array  # (H, P, 2)
    D: Array  # (H,)
    steps: Array  # (P,) — pre-sigmoid log-steps

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        *,
        key: PRNGKeyArray,
        r_min: float = 0.9,
        theta_max: float = jnp.pi,
    ):
        A_k, G_k, B_k, C_k, D_k, s_k = jr.split(key, 6)
        P, H = state_dim, input_dim

        self.steps = normal(stddev=0.5)(s_k, (P,))
        steps = nn.sigmoid(self.steps)

        # Initialise magnitudes in [r_min, 1] for stability
        mags = jnp.sqrt(
            random.uniform(G_k, (P,)) * (1.0 - r_min**2) + r_min**2
        )
        self.G_diag = (1 - mags**2) / (steps * mags**2)
        G = nn.relu(self.G_diag)

        theta = random.uniform(A_k, (P,)) * theta_max
        self.A_diag = _theta_to_A(theta, G, steps)

        std_B = 1.0 / math.sqrt(H)
        self.B = random.uniform(B_k, (P, H, 2)) * 2 * std_B - std_B
        std_C = 1.0 / math.sqrt(P)
        self.C = random.uniform(C_k, (H, P, 2)) * 2 * std_C - std_C
        self.D = normal(stddev=1.0)(D_k, (H,))

    def __call__(
        self, x: Array, key: PRNGKeyArray
    ) -> Array:  # x: (T, H) → (T, H)
        steps = nn.sigmoid(self.steps)
        G = nn.relu(self.G_diag)

        # Clamp A to valid stability region
        lo = (2 + steps * G - 2 * jnp.sqrt(1 + steps * G)) / steps**2
        hi = (2 + steps * G + 2 * jnp.sqrt(1 + steps * G)) / steps**2
        A = lo + nn.relu(self.A_diag - lo) - nn.relu(self.A_diag - hi)

        Bc = self.B[..., 0] + 1j * self.B[..., 1]
        Cc = self.C[..., 0] + 1j * self.C[..., 1]

        ys = _damped_linoss_imex(A, G, Bc, x, steps)  # (T, P) complex
        Cy = jax.vmap(lambda y: (Cc @ y).real)(ys)  # (T, H)
        Du = jax.vmap(lambda u: self.D * u)(x)  # (T, H)
        return Cy + Du


class StandardCausalConv1d(eqx.Module):
    """Standard (non-depthwise) causal 1-D convolution."""

    conv: eqx.nn.Conv1d
    kernel_size: int = eqx.field(static=True)

    def __init__(self, channels: int, kernel_size: int, *, key: PRNGKeyArray):
        self.kernel_size = kernel_size
        self.conv = eqx.nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            use_bias=False,
            key=key,
        )

    def __call__(self, x: Array) -> Array:
        # x is (T, C). Conv1d expects (C, T)
        x_t = x.T
        # pad left by kernel_size - 1 for causality
        x_pad = jnp.pad(x_t, ((0, 0), (self.kernel_size - 1, 0)))
        y = self.conv(x_pad)
        return y.T


class CausalBottleneck1D(eqx.Module):
    """Bottleneck causal convolution block using standard convolutions."""

    proj1: eqx.nn.Linear
    norm1: eqx.nn.LayerNorm
    conv2: StandardCausalConv1d
    norm2: eqx.nn.LayerNorm
    proj3: eqx.nn.Linear
    norm3: eqx.nn.LayerNorm

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        expansion: int = 2,
        *,
        key: PRNGKeyArray,
    ):
        k1, k2, k3 = jr.split(key, 3)
        mid_channels = channels * expansion

        self.proj1 = eqx.nn.Linear(
            channels, mid_channels, use_bias=False, key=k1
        )
        self.norm1 = eqx.nn.LayerNorm(mid_channels)

        # Replaced with standard causal conv!
        self.conv2 = StandardCausalConv1d(
            mid_channels, kernel_size=kernel_size, key=k2
        )
        self.norm2 = eqx.nn.LayerNorm(mid_channels)

        self.proj3 = eqx.nn.Linear(
            mid_channels, channels, use_bias=False, key=k3
        )
        self.norm3 = eqx.nn.LayerNorm(channels)

    def __call__(self, x: Array, state: eqx.nn.State, key: PRNGKeyArray):
        identity = x

        x = jax.vmap(self.proj1)(x)
        x, state = jax.vmap(self.norm1)(x, state)
        x = jax.nn.gelu(x)

        x = self.conv2(x)
        x, state = jax.vmap(self.norm2)(x, state)
        x = jax.nn.gelu(x)

        x = jax.vmap(self.proj3)(x)
        x, state = jax.vmap(self.norm3)(x, state)

        return identity + x, state


class PoolingLinossHeavyBlock(eqx.Module):
    """A LinOSS block with a pooling layer and channel expansion.

    Args:
        conv_position: Where to place the causal bottleneck convolutions
            relative to the sequence mixer. ``"pre"`` applies them before
            the mixer; ``"post"`` applies them after the mixer (but still
            inside the residual branch, before the skip-add).
    """

    norm: eqx.nn.LayerNorm
    mixer: LinOSSMixer
    glu: GLU
    drop: eqx.nn.Dropout
    res_blocks: list[CausalBottleneck1D]
    pool: eqx.nn.MaxPool1d
    expand: eqx.nn.Linear
    conv_position: str = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        state_dim: int,
        dropout_rate: float,
        num_res_blocks: int = 2,
        conv_kernel_size: int = 3,
        conv_position: str = "pre",
        *,
        key: PRNGKeyArray,
        r_min: float = 0.9,
        theta_max: float = jnp.pi,
    ):
        sm_k, glu_k, exp_k, *res_ks = jr.split(key, 3 + num_res_blocks)
        self.norm = eqx.nn.LayerNorm(in_dim)
        self.mixer = LinOSSMixer(
            in_dim, state_dim, key=sm_k, r_min=r_min, theta_max=theta_max
        )
        self.glu = GLU(in_dim, in_dim, key=glu_k)
        self.drop = eqx.nn.Dropout(p=dropout_rate)
        self.res_blocks = [
            CausalBottleneck1D(
                in_dim, kernel_size=conv_kernel_size, expansion=2, key=bk
            )
            for bk in res_ks
        ]
        self.pool = eqx.nn.MaxPool1d(kernel_size=2, stride=2)
        # Linear layer to double channels after pooling
        self.expand = eqx.nn.Linear(in_dim, out_dim, key=exp_k)
        self.conv_position = conv_position

    def __call__(self, x: Array, state: eqx.nn.State, key: PRNGKeyArray):
        d1, d2 = jr.split(key)
        skip = x

        if self.conv_position == "pre":
            for res_block in self.res_blocks:
                x, state = res_block(x, state, key)

        x = self.mixer(x, key=key)
        x, state = jax.vmap(self.norm)(x, state)
        x = self.drop(jax.nn.gelu(x), key=d1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=d2)

        if self.conv_position == "post":
            for res_block in self.res_blocks:
                x, state = res_block(x, state, key)

        y = skip + x

        # MaxPool expects (channels, length), we have (length, channels)
        y = y.T
        y = self.pool(y)
        y = y.T

        # Expand channel dimension just like strided Resnet blocks do
        y = jax.vmap(self.expand)(y)

        return y, state


class PoolingLinossHeavyBackbone(eqx.Module):
    encoder: eqx.nn.Linear
    blocks: list[PoolingLinossHeavyBlock]
    hidden_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        dropout_rate: float,
        state_dim: int,
        num_res_blocks: int = 2,
        conv_kernel_size: int = 3,
        conv_position: str = "pre",
        *,
        key: PRNGKeyArray,
        r_min: float = 0.9,
        theta_max: float = jnp.pi,
    ):
        self.hidden_dim = hidden_dim
        enc_k, *blk_ks = jr.split(key, num_blocks + 1)
        self.encoder = eqx.nn.Linear(input_dim, hidden_dim, key=enc_k)

        self.blocks = []
        current_dim = hidden_dim
        for bk in blk_ks:
            next_dim = current_dim * 2
            self.blocks.append(
                PoolingLinossHeavyBlock(
                    in_dim=current_dim,
                    out_dim=next_dim,
                    state_dim=state_dim,
                    dropout_rate=dropout_rate,
                    num_res_blocks=num_res_blocks,
                    conv_kernel_size=conv_kernel_size,
                    conv_position=conv_position,
                    key=bk,
                    r_min=r_min,
                    theta_max=theta_max,
                )
            )
            current_dim = next_dim

    def __call__(self, x: Array, state: eqx.nn.State, key: PRNGKeyArray):
        blk_ks = jr.split(key, len(self.blocks))
        y = jax.vmap(self.encoder)(x)
        for block, bk in zip(self.blocks, blk_ks, strict=True):
            y, state = block(y, state, bk)
        return y, state


class LinOSS(eqx.Module):
    time_linoss: PoolingLinossHeavyBackbone
    resnet: ResNet1D
    mlp: eqx.nn.MLP

    def __init__(
        self,
        time_in_features: int,
        time_hidden_dim: int,
        time_num_blocks: int,
        time_dropout_rate: float,
        time_state_dim: int,
        time_r_min: float,
        time_theta_max: float,
        time_num_res_blocks: int,
        time_conv_kernel_size: int,
        time_conv_position: str,
        resnet_layers: tuple[int, ...],
        resnet_latent_dim: int,
        resnet_kernel_size: int,
        resnet_norm_groups: int,
        mlp_width: int,
        mlp_depth: int,
        d_output: int = 1,
        *,
        key: PRNGKeyArray,
    ):
        k_time, k_res, k_mlp = jr.split(key, 3)
        self.time_linoss = PoolingLinossHeavyBackbone(
            input_dim=time_in_features,
            hidden_dim=time_hidden_dim,
            num_blocks=time_num_blocks,
            dropout_rate=time_dropout_rate,
            state_dim=time_state_dim,
            r_min=time_r_min,
            theta_max=time_theta_max,
            num_res_blocks=time_num_res_blocks,
            conv_kernel_size=time_conv_kernel_size,
            conv_position=time_conv_position,
            key=k_time,
        )

        final_time_dim = time_hidden_dim * (2**time_num_blocks)

        self.resnet = ResNet1D(
            in_channels=final_time_dim,
            layers=resnet_layers,
            classes=resnet_latent_dim,
            kernel_size=resnet_kernel_size,
            norm_groups=resnet_norm_groups,
            key=k_res,
        )
        self.mlp = eqx.nn.MLP(
            in_size=resnet_latent_dim,
            out_size=d_output,
            width_size=mlp_width,
            depth=mlp_depth,
            key=k_mlp,
        )

    def __call__(
        self,
        x_time: Array,
        state: eqx.nn.State,
        key: PRNGKeyArray,
    ) -> tuple[Array, eqx.nn.State]:
        x_time = x_time.T

        y_time, state = self.time_linoss(x_time, state, key)

        y_time = y_time.T

        y_res = self.resnet(y_time)

        logits = self.mlp(y_res)
        return logits, state
