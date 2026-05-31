import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.cli import LRSchedulerCallable
from numpy.typing import ArrayLike

from train.model.base import AframeBase
from train.metrics import TimeSlideAUROC
from train.utils.beta_nll_loss import BetaNLLLoss


def _log_gaussian_nll(
    task: pl.LightningModule,
    stage: str,
    nll: float,
    indiv_mse: ArrayLike,
    variance: ArrayLike,
) -> None:
    task.log(f"{stage}/gaussnll", nll, on_step=False, on_epoch=True, prog_bar=True)
    for i in range(len(indiv_mse)):
        task.log(f"{stage}/mse/out_{i}", indiv_mse[i], on_step=False, on_epoch=True)
        task.log(
            f"{stage}/sigma_{i}",
            torch.sqrt(variance[:, i].mean(dim=0)),
            on_step=False,
            on_epoch=True,
        )


def _log_within_percentile(
    task: pl.LightningModule,
    stage: str,
    mean_norm: torch.Tensor,
    y_target: torch.Tensor,
) -> None:
    y_target = y_target.reshape_as(mean_norm)
    mean_phys = mean_norm * task.y_std + task.y_mean
    rel_err = (mean_phys - y_target).abs() / y_target.abs().clamp(min=1e-8)
    for pct in [1, 2, 5, 10]:
        within = (rel_err < pct / 100.0).float()
        for i in range(within.shape[-1]):
            task.log(
                f"{stage}/within_{pct}pct/out_{i}",
                within[:, i].mean(),
                on_step=False,
                on_epoch=True,
            )


class RegressionAframe(AframeBase):
    """AframeBase + GaussianNLL loss, regression validation, and warmup+cosine optimizer.

    This is the base for all regression models (``LitS4DGaussianNLL``,
    ``LitLinOSSGaussianNLL``, etc.).  Classification models inherit from
    ``ClassificationAframe`` instead.

    Follows the same ``arch``-first convention as ``ClassificationAframe``:
    the architecture is pre-built and passed in; this class owns the
    training loop, loss, and optimizer logic.
    """

    def __init__(
        self,
        arch,
        d_output: int,
        learning_rate: float,
        weight_decay: float,
        metric: TimeSlideAUROC,
        warmup_steps: int = 1000,
        beta_nll: float = 0.5,
        lambda_spread: float = 0.0,
        y_mean: list[float] | None = None,
        y_std: list[float] | None = None,
        normalize_input: bool = False,
    ) -> None:
        super().__init__(arch)
        self.metric = metric
        if d_output % 2 != 0:
            raise ValueError(
                f"d_output={d_output} must be even (n_vars means + n_vars variances)."
            )
        self.n_vars = d_output // 2
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.lambda_spread = lambda_spread
        self.normalize_input = normalize_input
        self.criterion = BetaNLLLoss(beta=beta_nll)
        self.var_activation = nn.Softplus()

        _y_mean = (
            torch.tensor(y_mean, dtype=torch.float32)
            if y_mean is not None
            else torch.zeros(self.n_vars)
        )
        _y_std = (
            torch.tensor(y_std, dtype=torch.float32)
            if y_std is not None
            else torch.ones(self.n_vars)
        )
        self.register_buffer("y_mean", _y_mean)
        self.register_buffer("y_std", _y_std)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    def score(self, X: torch.Tensor) -> torch.Tensor:
        """Detection score: negative mean predicted variance (lower uncertainty → higher score)."""
        outputs = self(self._prepare_input(X))
        _, var_pre = outputs.chunk(2, dim=-1)
        return -self.var_activation(var_pre).mean(dim=-1)

    def _prepare_input(self, X: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            X = X / X.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return X

    def _normalize_target(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean) / self.y_std

    def _unnormalize_output(
        self, mean: torch.Tensor, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return mean * self.y_std + self.y_mean, sigma * self.y_std

    def m1_m2_to_chirp_mass(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)

    def compute_loss(self, batch):
        X, labels, params = batch

        outputs = self(self._prepare_input(X))
        mean, var = outputs.chunk(2, dim=-1)
        var = self.var_activation(var)

        chirp_mass = self.m1_m2_to_chirp_mass(params["mass_1"], params["mass_2"])
        y_norm = self._normalize_target(chirp_mass).reshape(mean.shape)

        indiv_mse = nn.MSELoss(reduction="none")(mean, y_norm).mean(dim=0)
        nll = self.criterion(mean, y_norm, var)
        spread = F.softplus(y_norm.detach().var(dim=0) - mean.var(dim=0)).mean()
        loss = nll + self.lambda_spread * spread

        return loss, nll, spread, indiv_mse, var, mean

    def training_step(self, batch, batch_idx):
        loss, nll, spread, indiv_mse, var, _ = self.compute_loss(batch)
        _log_gaussian_nll(self, "train", nll, indiv_mse, var)
        self.log("train/spread_penalty", spread)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        shift, X_bg, X_sig, params = batch

        y_bg = self.score(X_bg)

        n_views = X_sig.shape[0]
        all_loss, all_nll, all_spread, all_indiv_mse, all_var, all_mean_norm = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        all_scores_fg = []
        for i in range(n_views):
            loss, nll, spread, indiv_mse, var, mean_norm = self.compute_loss(
                (X_sig[i], None, params)
            )
            all_loss.append(loss)
            all_nll.append(nll)
            all_spread.append(spread)
            all_indiv_mse.append(indiv_mse)
            all_var.append(var)
            all_mean_norm.append(mean_norm)
            all_scores_fg.append(-var.mean(dim=-1))

        y_fg = torch.stack(all_scores_fg).mean(dim=0)
        self.metric.update(shift, y_bg, y_fg)
        self.log(
            "val/valid_auroc",
            self.metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        loss = torch.stack(all_loss).mean()
        nll = torch.stack(all_nll).mean()
        spread = torch.stack(all_spread).mean()
        indiv_mse = torch.stack(all_indiv_mse).mean(dim=0)
        var = torch.stack(all_var).mean(dim=0)

        # (n_views, batch, n_vars)
        mean_norm_views = torch.stack(all_mean_norm)
        mean_norm = mean_norm_views.mean(dim=0)
        view_variance = mean_norm_views.var(dim=0, correction=0)  # (batch, n_vars)

        chirp_mass = self.m1_m2_to_chirp_mass(params["mass_1"], params["mass_2"])

        _log_gaussian_nll(self, "val", nll, indiv_mse, var)
        _log_within_percentile(self, "val", mean_norm, chirp_mass)
        self.log("val/spread_penalty", spread, on_step=False, on_epoch=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        for i in range(view_variance.shape[-1]):
            self.log(
                f"val/view_var/out_{i}",
                view_variance[:, i].mean(),
                on_step=False,
                on_epoch=True,
            )

        mean_phys, sigma_phys = self._unnormalize_output(mean_norm, torch.sqrt(var))
        return {
            "targets": chirp_mass.detach().cpu(),
            "outputs": mean_phys.detach().cpu(),
            "params": {"snr": params["snr"].detach().cpu()},
            "all_outputs": {"chirp_mass_std": sigma_phys.detach().cpu()},
        }

    def test_step(self, batch, batch_idx):
        X, y_target, _ = batch
        outputs = self(self._prepare_input(X))
        mean_norm = outputs[:, : self.n_vars]
        sigma_norm = torch.sqrt(self.var_activation(outputs[:, self.n_vars :]))
        mean_phys, sigma_phys = self._unnormalize_output(mean_norm, sigma_norm)
        return {
            "y_true": y_target.detach().cpu(),
            "y_pred": mean_phys.detach().cpu(),
            "y_sigma": sigma_phys.detach().cpu(),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-2,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - self.warmup_steps)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.warmup_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class RegressionAframeS4D(RegressionAframe):
    """S4D sequence model trained with GaussianNLL for parameter estimation.

    Pass a pre-built ``S4Model`` (or compatible) as ``arch``.
    ``d_output`` must be even: first half = means, second half = pre-Softplus variances.

    Uses S4D-aware optimizer: SSM parameters (those with ``._optim``) get their
    own learning-rate group; all other parameters share ``base_lr``.
    """

    def __init__(
        self,
        arch,
        d_output: int,
        metric: TimeSlideAUROC,
        base_lr: float = 1e-4,
        weight_decay: float = 0.0,
        warmup_steps: int = 1000,
        beta_nll: float = 0.5,
        lambda_spread: float = 0.0,
        lr_scheduler: LRSchedulerCallable | None = None,
        lr_scheduler_interval: str = "epoch",
        y_mean: list[float] | None = None,
        y_std: list[float] | None = None,
        normalize_input: bool = False,
        log_gradients: bool = False,
    ) -> None:
        super().__init__(
            arch,
            d_output=d_output,
            metric=metric,
            learning_rate=base_lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            beta_nll=beta_nll,
            lambda_spread=lambda_spread,
            y_mean=y_mean,
            y_std=y_std,
            normalize_input=normalize_input,
        )
        self._lr_scheduler_factory = lr_scheduler
        self.log_gradients = log_gradients
        self.save_hyperparameters(ignore=["arch", "lr_scheduler", "metric"])

    def on_after_backward(self) -> None:
        if self.log_gradients:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.log(
                        f"grad_norm/{name}",
                        param.grad.norm(),
                        on_step=True,
                        on_epoch=False,
                    )
                if "log_A_real" in name:
                    self.log(
                        f"ssm/A_real_mean/{name}",
                        -param.exp().mean(),
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"ssm/A_real_max/{name}",
                        -param.exp().max(),
                        on_step=False,
                        on_epoch=True,
                    )
                if "log_dt" in name:
                    self.log(
                        f"ssm/dt_mean/{name}",
                        param.exp().mean(),
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"ssm/dt_max/{name}",
                        param.exp().max(),
                        on_step=False,
                        on_epoch=True,
                    )

    def configure_optimizers(self):
        hp = self.hparams
        all_params = list(self.model.parameters())
        default_params = [p for p in all_params if not hasattr(p, "_optim")]
        optim_params = [p for p in all_params if hasattr(p, "_optim")]
        param_groups = [
            {
                "params": default_params,
                "lr": hp.base_lr,
                "weight_decay": hp.weight_decay,
            }
        ]
        unique_hps = [
            dict(s)
            for s in sorted(set(frozenset(p._optim.items()) for p in optim_params))
        ]
        for ohp in unique_hps:
            group = {
                "params": [p for p in optim_params if getattr(p, "_optim") == ohp],
                "lr": ohp.get("lr", hp.base_lr),
            }
            group.update(ohp)
            param_groups.append(group)
        optimizer = torch.optim.AdamW(param_groups)
        if self._lr_scheduler_factory is None:
            return optimizer
        scheduler = self._lr_scheduler_factory(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": hp.lr_scheduler_interval,
            },
        }


class JaxRegressionAframe(AframeBase):
    """LinOSS (JAX/equinox) regression model trained with BetaNLL loss.

    Uses optax for optimization (``automatic_optimization=False``) since
    JAX models cannot be differentiated through PyTorch autograd.

    Pass a pre-built ``RegressionTimeDomainLinOSS`` (or any ``JaxArchitecture``
    whose ``__call__`` returns ``(outputs, state)`` with ``outputs`` shape
    ``(d_output,)`` per sample) as ``arch``.
    """

    def __init__(
        self,
        arch,
        d_output: int,
        metric: TimeSlideAUROC,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        warmup_steps: int = 1000,
        max_steps: int = 500_000,
        clip_grad_norm: float = 10.0,
        beta_nll: float = 0.5,
        lambda_spread: float = 0.0,
        y_mean: list[float] | None = None,
        y_std: list[float] | None = None,
        normalize_input: bool = False,
        seed: int = 42,
    ) -> None:
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import jax.random as jr
        import optax
        from architectures.base import Architecture

        super().__init__(arch=Architecture())  # dummy PyTorch module
        self.automatic_optimization = False
        self.metric = metric

        if d_output % 2 != 0:
            raise ValueError(
                f"d_output={d_output} must be even (n_vars means + n_vars log-vars)."
            )
        self.n_vars = d_output // 2
        self.beta_nll_coef = beta_nll
        self.lambda_spread = lambda_spread
        self.normalize_input = normalize_input
        self.save_hyperparameters(ignore=["arch", "metric"])

        # JAX model + equinox state
        self.jax_model = arch
        self.jax_model_state = eqx.nn.State(self.jax_model)
        self.jax_model_filter_spec = jax.tree_util.tree_map(
            eqx.is_inexact_array, self.jax_model
        )

        # optax optimizer: warmup → cosine decay
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=max_steps,
            end_value=learning_rate * 0.01,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(clip_grad_norm),
            optax.inject_hyperparams(optax.adamw)(
                learning_rate=scheduler, weight_decay=weight_decay
            ),
        )
        diff_model, _ = eqx.partition(self.jax_model, self.jax_model_filter_spec)
        self.opt_state = self.optimizer.init(diff_model)

        self.y_mean = torch.tensor(
            y_mean if y_mean is not None else [0.0] * self.n_vars,
            dtype=torch.float32,
        )
        self.y_std = torch.tensor(
            y_std if y_std is not None else [1.0] * self.n_vars,
            dtype=torch.float32,
        )
        self.rng_key = jr.PRNGKey(seed)

    def configure_optimizers(self):
        pass  # optax manages all optimisation

    def _to_jax(self, t: torch.Tensor):
        import jax.numpy as jnp
        return jnp.asarray(t.cpu().numpy())

    def _normalize_target(self, cm):
        import jax.numpy as jnp
        mean = jnp.array(self.y_mean.numpy())
        std = jnp.array(self.y_std.numpy())
        return (cm - mean) / std

    @staticmethod
    def _chirp_mass_jax(m1, m2):
        return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)

    def training_step(self, batch, batch_idx):
        import jax.numpy as jnp
        import jax.random as jr
        from train.utils.jax.training import jax_apply_regression_training_step

        X, _, params = batch
        X_j = self._to_jax(X)
        if self.normalize_input:
            X_j = X_j / jnp.std(X_j, axis=-1, keepdims=True).clip(1e-8)

        cm = self._chirp_mass_jax(
            self._to_jax(params["mass_1"]), self._to_jax(params["mass_2"])
        )
        cm_norm = self._normalize_target(cm).reshape(-1, self.n_vars)

        self.rng_key, k = jr.split(self.rng_key)
        keys = jr.split(k, X_j.shape[0])

        (
            self.jax_model,
            self.jax_model_state,
            self.opt_state,
            metrics,
        ) = jax_apply_regression_training_step(
            self.jax_model,
            self.jax_model_filter_spec,
            self.jax_model_state,
            X_j,
            cm_norm,
            float(self.beta_nll_coef),
            float(self.lambda_spread),
            self.opt_state,
            self.optimizer.update,
            keys,
        )

        loss = float(metrics["loss"])
        nll = float(metrics["nll"])
        spread = float(metrics["spread"])
        var_np = np.array(metrics["var"])  # (B, n_vars)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/gaussnll", nll, on_step=False, on_epoch=True)
        self.log("train/spread_penalty", spread, on_step=False, on_epoch=True)
        for i in range(self.n_vars):
            self.log(
                f"train/sigma_{i}",
                float(np.sqrt(var_np[:, i].mean())),
                on_step=False,
                on_epoch=True,
            )
        self.log(
            "train/lr",
            float(self.opt_state[1].hyperparams["learning_rate"]),
            on_step=True,
            on_epoch=True,
        )

        # Keep Lightning's manual-optimisation bookkeeping happy
        optimizers = self.optimizers()
        if isinstance(optimizers, (list, tuple)):
            for opt in optimizers:
                opt.step()
        elif optimizers is not None:
            optimizers.step()

        return torch.tensor(0.0)

    def _jax_inference(self, X: torch.Tensor) -> np.ndarray:
        """JAX inference → numpy array (B, d_output)."""
        import jax.numpy as jnp
        import jax.random as jr
        from train.utils.jax.training import jax_inference

        X_j = self._to_jax(X)
        if self.normalize_input:
            X_j = X_j / jnp.std(X_j, axis=-1, keepdims=True).clip(1e-8)
        self.rng_key, k = jr.split(self.rng_key)
        keys = jr.split(k, X_j.shape[0])
        outputs, new_state = jax_inference(
            self.jax_model, X_j, self.jax_model_state, keys
        )
        self.jax_model_state = new_state
        return np.array(outputs, copy=True)

    def score(self, X: torch.Tensor) -> torch.Tensor:
        """Detection score: negative mean predicted variance."""
        outputs = self._jax_inference(X)  # (B, d_output)
        var_pre = outputs[:, self.n_vars:]
        var = np.log1p(np.exp(var_pre))  # softplus in numpy
        return torch.tensor(-var.mean(axis=-1))

    def validation_step(self, batch, batch_idx):
        shift, X_bg, X_sig, params = batch

        y_bg = self.score(X_bg)

        cm_torch = (
            (params["mass_1"] * params["mass_2"]) ** (3 / 5)
            / (params["mass_1"] + params["mass_2"]) ** (1 / 5)
        ).cpu()

        n_views = X_sig.shape[0]
        all_nll, all_mean_norm, all_var, all_scores_fg = [], [], [], []

        for i in range(n_views):
            outputs = self._jax_inference(X_sig[i])  # (B, d_output) numpy
            mean_np = outputs[:, : self.n_vars]
            var_pre_np = outputs[:, self.n_vars :]
            var_np = np.log1p(np.exp(var_pre_np))  # softplus

            cm_norm_np = np.array(
                self._normalize_target(
                    self._to_jax(cm_torch).reshape(-1, self.n_vars)
                )
            )
            nll_val = float(
                np.mean(0.5 * (np.log(var_np) + (cm_norm_np - mean_np) ** 2 / var_np))
            )
            all_nll.append(nll_val)
            all_mean_norm.append(mean_np)
            all_var.append(var_np)
            all_scores_fg.append(-var_np.mean(axis=-1))

        y_fg = torch.tensor(np.stack(all_scores_fg).mean(axis=0))
        self.metric.update(shift, y_bg, y_fg)
        self.log(
            "val/valid_auroc",
            self.metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        nll = float(np.mean(all_nll))
        var_stack = np.stack(all_var).mean(axis=0)        # (B, n_vars)
        mean_norm_views = np.stack(all_mean_norm)          # (n_views, B, n_vars)
        mean_norm_stack = mean_norm_views.mean(axis=0)     # (B, n_vars)
        view_variance = np.var(mean_norm_views, axis=0, ddof=0)  # (B, n_vars)

        cm_norm_np = np.array(
            self._normalize_target(
                self._to_jax(cm_torch).reshape(-1, self.n_vars)
            )
        )

        self.log("val/gaussnll", nll, on_step=False, on_epoch=True, prog_bar=True)
        for i in range(self.n_vars):
            mse_i = float(np.mean((mean_norm_stack[:, i] - cm_norm_np[:, i]) ** 2))
            sigma_i = float(np.sqrt(var_stack[:, i].mean()))
            self.log(f"val/mse/out_{i}", mse_i, on_step=False, on_epoch=True)
            self.log(f"val/sigma_{i}", sigma_i, on_step=False, on_epoch=True)
            self.log(
                f"val/view_var/out_{i}",
                float(view_variance[:, i].mean()),
                on_step=False,
                on_epoch=True,
            )

        mean_norm_t = torch.tensor(mean_norm_stack, dtype=torch.float32)
        mean_phys = mean_norm_t * self.y_std + self.y_mean
        cm_target = cm_torch.reshape_as(mean_phys)
        rel_err = (mean_phys - cm_target).abs() / cm_target.abs().clamp(min=1e-8)
        for pct in [1, 2, 5, 10]:
            for i in range(self.n_vars):
                self.log(
                    f"val/within_{pct}pct/out_{i}",
                    (rel_err[:, i] < pct / 100.0).float().mean(),
                    on_step=False,
                    on_epoch=True,
                )

        sigma_phys = torch.tensor(np.sqrt(var_stack), dtype=torch.float32) * self.y_std
        return {
            "targets": cm_torch.detach().cpu(),
            "outputs": mean_phys.detach().cpu(),
            "params": {"snr": params["snr"].detach().cpu()},
            "all_outputs": {"chirp_mass_std": sigma_phys.detach().cpu()},
        }
