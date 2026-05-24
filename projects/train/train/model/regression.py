import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.cli import LRSchedulerCallable
from numpy.typing import ArrayLike


class WarmupCosineAnnealingWarmRestarts(torch.optim.lr_scheduler.LRScheduler):
    """Linear warmup followed by CosineAnnealingWarmRestarts (epoch-based)."""

    def __init__(self, optimizer, warmup_epochs, T_0, T_mult=2, eta_min=1e-8,
                 warmup_start_factor=0.01, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_start_factor = warmup_start_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            alpha = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * e / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        t = e - self.warmup_epochs
        T_cur, T_i = self._cosine_position(t)
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
                for base_lr in self.base_lrs]

    def _cosine_position(self, t):
        T_i = self.T_0
        while t >= T_i:
            t -= T_i
            T_i *= self.T_mult
        return t, T_i


class BetaNLLLoss(nn.Module):
    """β-NLL loss (Seitzer et al. 2022, https://arxiv.org/abs/2203.09168).

    Standard GaussianNLL has a degenerate minimum where inflating var drives
    the mean gradient to zero.  β-weighting prevents this:

        L_β = sg(var)^β · L_NLL
        ∂L_β/∂mean = sg(var)^(β−1) · (mean − y)

    β=0 → standard NLL (degenerate); β=0.5 → recommended; β=1 → MSE gradient.
    """

    def __init__(self, beta: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f'beta must be in [0, 1], got {beta}')
        self.beta = beta
        self.reduction = reduction

    def forward(self, mean: torch.Tensor, target: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        nll = 0.5 * (torch.log(var) + (mean - target) ** 2 / var)
        if self.beta > 0.0:
            nll = nll * var.detach().pow(self.beta)
        return nll.mean() if self.reduction == 'mean' else nll.sum()


def _log_gaussian_nll(
    task: pl.LightningModule,
    stage: str,
    nll: float,
    indiv_mse: ArrayLike,
    variance: ArrayLike,
) -> None:
    task.log(f'{stage}/gaussnll', nll, on_step=False, on_epoch=True, prog_bar=True)
    for i in range(len(indiv_mse)):
        task.log(f'{stage}/mse/out_{i}', indiv_mse[i], on_step=False, on_epoch=True)
        task.log(f'{stage}/sigma_{i}', torch.sqrt(variance[i].mean(dim=0)), on_step=False, on_epoch=True)


def _log_within_pct(
    task: pl.LightningModule,
    stage: str,
    mean_norm: torch.Tensor,
    y_target: torch.Tensor,
) -> None:
    mean_phys = mean_norm * task.y_std + task.y_mean
    rel_err = (mean_phys - y_target).abs() / y_target.abs().clamp(min=1e-8)
    for pct in [1, 2, 5, 10]:
        within = (rel_err < pct / 100.0).float()
        for i in range(within.shape[-1]):
            task.log(f'{stage}/within_{pct}pct/out_{i}', within[:, i].mean(), on_step=False, on_epoch=True)


class _RegressionBase(pl.LightningModule):
    """Shared GaussianNLL loss, logging, and step logic for regression tasks."""

    def __init__(
        self,
        d_output: int,
        learning_rate: float,
        weight_decay: float,
        warmup_steps: int = 1000,
        beta_nll: float = 0.5,
        lambda_spread: float = 0.0,
        y_mean: list[float] | None = None,
        y_std: list[float] | None = None,
        normalize_input: bool = False,
    ) -> None:
        super().__init__()
        if d_output % 2 != 0:
            raise ValueError(f'd_output={d_output} must be even (n_vars means + n_vars variances).')
        self.n_vars = d_output // 2
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.lambda_spread = lambda_spread
        self.normalize_input = normalize_input
        self.criterion = BetaNLLLoss(beta=beta_nll)
        self.var_activation = nn.Softplus()

        # Output normalization buffers — saved in checkpoint, auto-moved to device.
        # Model trains in normalized space; inference un-normalizes for physical outputs.
        _y_mean = torch.tensor(y_mean, dtype=torch.float32) if y_mean is not None else torch.zeros(self.n_vars)
        _y_std = torch.tensor(y_std, dtype=torch.float32) if y_std is not None else torch.ones(self.n_vars)
        self.register_buffer('y_mean', _y_mean)
        self.register_buffer('y_std', _y_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _prepare_input(self, X_sequence: torch.Tensor) -> torch.Tensor:
        """Normalize input if requested; subclasses may also transpose."""
        if self.normalize_input:
            # Per-sample per-channel standardization (whitened strain already has ~unit RMS,
            # but this makes each sample exactly unit std regardless of noise level)
            X_sequence = X_sequence / X_sequence.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return X_sequence

    def _normalize_target(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean) / self.y_std

    def _unnormalize_output(self, mean: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert normalized model outputs back to physical units."""
        return mean * self.y_std + self.y_mean, sigma * self.y_std

    def compute_loss(self, batch):
        X_sequence, y_target, _ = batch
        outputs = self(self._prepare_input(X_sequence))
        mean = outputs[:, :self.n_vars]
        var = self.var_activation(outputs[:, self.n_vars:])
        y_norm = self._normalize_target(y_target)
        indiv_mse = nn.MSELoss(reduction='none')(mean, y_norm).T.mean(dim=1)  # (n_vars,) in normalized space
        nll = self.criterion(mean, y_norm, var)
        spread = F.softplus(y_norm.detach().var(dim=0) - mean.var(dim=0)).mean()
        loss = nll + self.lambda_spread * spread
        return loss, nll, spread, indiv_mse, var, mean, y_target

    def training_step(self, batch, batch_idx):
        loss, nll, spread, indiv_mse, var, _, _ = self.compute_loss(batch)
        _log_gaussian_nll(self, 'train', nll, indiv_mse, var)
        self.log('train/spread_penalty', spread, on_step=False, on_epoch=True)
        self.log('train/loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, nll, spread, indiv_mse, var, mean_norm, y_target = self.compute_loss(batch)
        _log_gaussian_nll(self, 'val', nll, indiv_mse, var)
        _log_within_pct(self, 'val', mean_norm, y_target)
        self.log('val/spread_penalty', spread, on_step=False, on_epoch=True)
        self.log('val/loss', loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        X_sequence, y_target, _ = batch
        outputs = self(self._prepare_input(X_sequence))
        mean_norm = outputs[:, :self.n_vars]
        sigma_norm = torch.sqrt(self.var_activation(outputs[:, self.n_vars:]))
        mean_phys, sigma_phys = self._unnormalize_output(mean_norm, sigma_norm)
        return {
            'y_true': y_target.detach().cpu(),
            'y_pred': mean_phys.detach().cpu(),
            'y_sigma': sigma_phys.detach().cpu(),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, end_factor=1.0, total_iters=self.warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - self.warmup_steps)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_steps]
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}


class LitS4DGaussianNLL(_RegressionBase):
    """S4D sequence model trained with GaussianNLLLoss for parameter estimation.

    Input batch: (X_sequence, y_target, z_observed)
      - X_sequence: (B, n_ifos, L) channels-first
      - y_target:   (B, n_vars)
      - z_observed: (B, n_obs)  not used in loss

    d_output must be even: first half means, second half pre-Softplus variances.
    """

    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.2,
        dt_min: float = 1e-3,
        dt_max: float = 1.0,
        lr: float | None = None,
        base_lr: float = 1e-4,
        weight_decay: float = 0.0,
        beta_nll: float = 0.5,
        lambda_spread: float = 0.0,
        lr_scheduler: LRSchedulerCallable | None = None,
        lr_scheduler_interval: str = 'epoch',
        y_mean: list[float] | None = None,
        y_std: list[float] | None = None,
        normalize_input: bool = False,
    ) -> None:
        super().__init__(d_output=d_output, learning_rate=base_lr, weight_decay=weight_decay,
                         beta_nll=beta_nll, lambda_spread=lambda_spread,
                         y_mean=y_mean, y_std=y_std, normalize_input=normalize_input)
        self._lr_scheduler_factory = lr_scheduler
        self.save_hyperparameters(ignore=['lr_scheduler'])
        self.model = None
        self.configure_model()

    def configure_model(self) -> None:
        if self.model is not None:
            return
        from architectures.networks.s4d import S4Model
        hp = self.hparams
        self.model = S4Model(
            d_input=hp.d_input, d_output=hp.d_output, d_model=hp.d_model,
            d_state=hp.d_state, n_layers=hp.n_layers, dropout=hp.dropout,
            dt_min=hp.dt_min, dt_max=hp.dt_max, lr=hp.lr,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_after_backward(self) -> None:
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.log(f'grad_norm/{name}', param.grad.norm(), on_step=True, on_epoch=False)
            if 'log_A_real' in name:
                self.log(f'ssm/A_real_mean/{name}', -param.exp().mean(), on_step=False, on_epoch=True)
                self.log(f'ssm/A_real_max/{name}', -param.exp().max(), on_step=False, on_epoch=True)
            if 'log_dt' in name:
                self.log(f'ssm/dt_mean/{name}', param.exp().mean(), on_step=False, on_epoch=True)
                self.log(f'ssm/dt_max/{name}', param.exp().max(), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        hp = self.hparams
        all_params = list(self.model.parameters())
        default_params = [p for p in all_params if not hasattr(p, '_optim')]
        optim_params = [p for p in all_params if hasattr(p, '_optim')]
        param_groups = [{'params': default_params, 'lr': hp.base_lr, 'weight_decay': hp.weight_decay}]
        unique_hps = [dict(s) for s in sorted(set(frozenset(p._optim.items()) for p in optim_params))]
        for ohp in unique_hps:
            group = {'params': [p for p in optim_params if getattr(p, '_optim') == ohp],
                     'lr': ohp.get('lr', hp.base_lr)}
            group.update(ohp)
            param_groups.append(group)
        optimizer = torch.optim.AdamW(param_groups)
        if self._lr_scheduler_factory is None:
            return optimizer
        scheduler = self._lr_scheduler_factory(optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': hp.lr_scheduler_interval}}


class LitLinOSSGaussianNLL(_RegressionBase):
    """LinOSS sequence model trained with GaussianNLLLoss for parameter estimation.

    Input batch: (X_sequence, y_target, z_observed)
      - X_sequence: (B, n_ifos, L) channels-first
      - y_target:   (B, n_vars)
      - z_observed: (B, n_obs)  not used in loss

    LinOSSModel expects (B, L, d_input), so the input is transposed in _prepare_input.
    """

    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int = 64,
        ssm_size: int = 64,
        n_layers: int = 4,
        dropout: float = 0.0,
        discretization: str = 'IM',
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        warmup_steps: int = 1000,
        beta_nll: float = 0.5,
        y_mean: list[float] | None = None,
        y_std: list[float] | None = None,
        normalize_input: bool = False,
    ) -> None:
        super().__init__(d_output=d_output, learning_rate=learning_rate, weight_decay=weight_decay,
                         warmup_steps=warmup_steps, beta_nll=beta_nll,
                         y_mean=y_mean, y_std=y_std, normalize_input=normalize_input)
        self.save_hyperparameters()
        self.model = None
        self.configure_model()

    def configure_model(self) -> None:
        if self.model is not None:
            return
        from architectures.networks.linoss import LinOSSModel
        hp = self.hparams
        self.model = LinOSSModel(
            d_input=hp.d_input, d_output=hp.d_output, d_model=hp.d_model,
            ssm_size=hp.ssm_size, n_layers=hp.n_layers, dropout=hp.dropout,
            discretization=hp.discretization,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _prepare_input(self, X_sequence: torch.Tensor) -> torch.Tensor:
        X_sequence = super()._prepare_input(X_sequence)
        # LinOSSModel expects (B, L, d_input); X_sequence arrives as (B, d_input, L)
        return X_sequence.transpose(1, 2)
