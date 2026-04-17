# SPDX-License-Identifier: Apache-2.0
#
# This file is a derivative work of the S4 repository:
#   https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py
#   https://github.com/state-spaces/s4/blob/main/src/models/nn/dropout.py
#
# Copyright (c) 2023 The S4 Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications copyright (c) 2021 Alec Gunny and aframe contributors
# This project (aframe) is released under the MIT License; see LICENSE.
#
# Modifications made 2024–2026:
#   - Inlined DropoutNd from src/models/nn/dropout.py (no external dep).
#   - Replaced hardcoded `length` buffer in S4DKernel with dynamic L passed
#     to forward().
#   - Replaced Python complex literal `1j` with `torch.complex(...)` for
#     torch.compile compatibility.
#   - Removed `prenorm` from S4Model (postnorm only).
#   - Removed `length` parameter from S4Model/S4D (inferred from input).
#   - Added PEP 604 type hints throughout (`float | None` etc.).

"""Minimal S4D implementation for aframe.

Architecture (S4Model):
    Linear encoder  (d_input → d_model)
    n_layers × S4D residual block  (postnorm)
    Mean pool over L
    Linear decoder  (d_model → d_output)

Input:  (B, d_input, L)   — channels-first, matching aframe convention.
Output: (B, d_output)
"""

import math

import torch
import torch.nn as nn
from einops import rearrange, repeat


class DropoutNd(nn.Module):
    """N-dimensional dropout that ties the mask across sequence positions."""

    def __init__(self, p: float = 0.5, tie: bool = True, transposed: bool = True):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(
                f"dropout probability must be in [0, 1), got {p}"
            )
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (batch, dim, lengths...) if transposed else (batch, lengths..., dim)."""
        if self.training:
            if not self.transposed:
                X = rearrange(X, "b ... d -> b d ...")
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            mask = torch.rand(*mask_shape, device=X.device) < 1.0 - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                X = rearrange(X, "b d ... -> b ... d")
        return X


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(
        self,
        d_model: int,
        N: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float | None = None,
    ):
        super().__init__()

        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), "n -> h n", h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L: int) -> torch.Tensor:
        """Returns: (H, L) convolution kernel."""
        dt = torch.exp(self.log_dt)               # (H,)
        C = torch.view_as_complex(self.C)         # (H, N//2)
        # torch.complex(...) instead of `1j` for torch.compile safety
        A = torch.complex(-torch.exp(self.log_A_real), self.A_imag)  # (H, N//2)

        dtA = A * dt.unsqueeze(-1)                                    # (H, N//2)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)     # (H, N//2, L)
        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum("hn, hnl -> hl", C, torch.exp(K)).real  # (H, L)
        return K

    def register(self, name: str, tensor: torch.Tensor, lr: float | None = None) -> None:
        """Register a tensor with a configurable learning rate and 0 weight decay."""
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            getattr(self, name)._optim = optim


class S4D(nn.Module):
    """Single S4D layer operating on (B, H, L) sequences."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        transposed: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float | None = None,
    ):
        super().__init__()
        self.transposed = transposed
        self.D = nn.Parameter(torch.randn(d_model))

        self.kernel = S4DKernel(d_model, N=d_state, dt_min=dt_min, dt_max=dt_max, lr=lr)

        self.activation = nn.GELU()
        self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u: torch.Tensor, **kwargs) -> tuple[torch.Tensor, None]:
        """
        Args:
            u: (B, H, L) if transposed else (B, L, H)

        Returns:
            (B, H, L) output and None (dummy state placeholder).
        """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        k = self.kernel(L=L)                               # (H, L)
        k_f = torch.fft.rfft(k, n=2 * L)                  # (H, L)
        u_f = torch.fft.rfft(u, n=2 * L)                  # (B, H, L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B, H, L)

        y = y + u * self.D.unsqueeze(-1)  # D-term skip connection
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y, None


class S4Model(nn.Module):
    """Full S4D sequence model for regression / classification.

    Input:  (B, d_input, L)  — channels-first (aframe convention).
    Output: (B, d_output)
    """

    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float | None = None,
    ):
        super().__init__()

        self.encoder = nn.Linear(d_input, d_model)

        self.s4_layers = nn.ModuleList([
            S4D(d_model, d_state=d_state, dropout=dropout, transposed=True,
                dt_min=dt_min, dt_max=dt_max, lr=lr)
            for _ in range(n_layers)
        ])
        self.norms    = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([DropoutNd(dropout)    for _ in range(n_layers)])

        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, d_input, L)

        Returns:
            (B, d_output)
        """
        x = x.transpose(-1, -2)  # (B, L, d_input)
        x = self.encoder(x)      # (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, d_model, L)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts, strict=True):
            z, _ = layer(x)
            z = dropout(z)
            x = norm((z + x).transpose(-1, -2)).transpose(-1, -2)  # postnorm

        x = x.transpose(-1, -2)  # (B, L, d_model)
        x = x.mean(dim=1)        # (B, d_model) — pool over sequence
        return self.decoder(x)   # (B, d_output)
