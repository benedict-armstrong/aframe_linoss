"""Regression data module using aframe's on-the-fly injection pipeline.

Architecture:
    Training (disk):      background HDF5 + waveform HDF5 → PSD estimation
                          → waveform projection/injection → whitening
                          → (X_whitened, params, empty_z)
    Training (generator): background HDF5 → PSD estimation → prior sample
                          → waveform generation → projection/injection
                          → whitening → (X_whitened, params, empty_z)
    Validation: same pipeline over held-out background + validation waveforms,
                paired with their chirp_mass labels.

Batch format throughout: (X, y_params, z_empty)
    X          : (B, n_ifos, L) whitened strain, channels-first
    y_params   : (B, n_target_params) e.g. chirp_mass
    z_empty    : (B, 0)  placeholder (no observed variables used)
"""

import math
import warnings
from pathlib import Path
from typing import Iterable

import h5py
import lightning.pytorch as pl
import numpy as np
import torch
from ml4gw.utils.slicing import sample_kernels

from train.data.base import BaseAframeDataset, ZippedDataset


def _compute_target_params(
    m1: np.ndarray,
    m2: np.ndarray,
    target_parameters: tuple[str, ...],
) -> np.ndarray:
    available = {
        "chirp_mass": ((m1 * m2) ** 3 / (m1 + m2)) ** (1 / 5),
        "mass_ratio": m2 / m1,
        "mass_1": m1,
        "mass_2": m2,
    }
    cols = []
    for name in target_parameters:
        if name not in available:
            raise ValueError(
                f"Unknown target parameter {name!r}. "
                f"Choose from {list(available)}"
            )
        cols.append(available[name])
    return np.stack(cols, axis=-1).astype(np.float32)


def _compute_target_params_tensor(
    parameters: dict,
    target_parameters: tuple[str, ...],
) -> torch.Tensor:
    """Same as _compute_target_params but from a prior sample dict of tensors."""
    available = {k: v for k, v in parameters.items()}
    if "chirp_mass" not in available and "mass_1" in available and "mass_2" in available:
        m1, m2 = available["mass_1"], available["mass_2"]
        available["chirp_mass"] = ((m1 * m2) ** 3 / (m1 + m2)) ** 0.2
        available["mass_ratio"] = m2 / m1
    elif "mass_1" not in available and "chirp_mass" in available and "mass_ratio" in available:
        mc, q = available["chirp_mass"], available["mass_ratio"]
        available["mass_1"] = mc * (1 + q) ** 0.2 / q ** 0.6
        available["mass_2"] = mc * q ** 0.4 * (1 + q) ** 0.2
    cols = []
    for name in target_parameters:
        if name not in available:
            raise ValueError(
                f"Unknown target parameter {name!r}. "
                f"Choose from {list(available)}"
            )
        cols.append(available[name])
    return torch.stack(cols, dim=-1).float()


class _WaveformParamLoader(torch.utils.data.IterableDataset):
    """Load waveform polarisations and intrinsic parameters from HDF5 files.

    Each iteration yields ``(polarizations, params)`` where:
        polarizations : (batch_size, 2, L)  — [cross, plus]
        params        : (batch_size, n_params)
    """

    def __init__(
        self,
        fnames: Iterable[Path],
        target_parameters: tuple[str, ...],
        batch_size: int,
        batches_per_epoch: int,
        chunk_size: int = 1000,
    ) -> None:
        self.fnames = list(fnames)
        self.target_parameters = target_parameters
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.chunk_size = chunk_size

        self.sizes: dict[Path, int] = {}
        self._pol_dsets: dict[Path, dict[str, h5py.Dataset]] = {}
        self._param_dsets: dict[Path, dict[str, h5py.Dataset]] = {}
        self._files: dict[Path, h5py.File] = {}

        for fname in self.fnames:
            f = h5py.File(fname, "r")
            self._files[fname] = f
            wf_grp = f["waveforms"]
            pm_grp = f["parameters"]

            self._pol_dsets[fname] = {ch: wf_grp[ch] for ch in ("cross", "plus")}
            self._param_dsets[fname] = {k: pm_grp[k] for k in ("mass_1", "mass_2")}

            n = len(self._pol_dsets[fname]["cross"])
            self.sizes[fname] = n

            if self._pol_dsets[fname]["cross"].chunks is None:
                warnings.warn(
                    f"{fname} lacks chunked storage — I/O may be slow.",
                    stacklevel=2,
                )

        self.waveform_size = self._pol_dsets[self.fnames[0]]["cross"].shape[1]
        self.probs = np.array(
            [self.sizes[f] / self.total for f in self.fnames]
        )

    @property
    def total(self) -> int:
        return sum(self.sizes.values())

    @property
    def chunks_per_batch(self) -> int:
        return math.ceil(self.batch_size / self.chunk_size)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __del__(self) -> None:
        for f in self._files.values():
            try:
                f.close()
            except Exception:
                pass

    def _load_chunk(self, fname: Path, start: int, size: int):
        end = min(start + size, self.sizes[fname])
        pol = {ch: self._pol_dsets[fname][ch][start:end] for ch in ("cross", "plus")}
        pm = {k: self._param_dsets[fname][k][start:end] for k in ("mass_1", "mass_2")}
        return pol, pm

    def sample_batch(self):
        batch_pol = np.zeros(
            (self.batch_size, 2, self.waveform_size), dtype=np.float32
        )
        m1_buf = np.zeros(self.batch_size, dtype=np.float64)
        m2_buf = np.zeros(self.batch_size, dtype=np.float64)

        for i in range(self.chunks_per_batch):
            fname = np.random.choice(self.fnames, p=self.probs)
            chunk_size = min(self.chunk_size, self.batch_size - i * self.chunk_size)
            max_start = self.sizes[fname] - chunk_size
            start = np.random.randint(0, max_start + 1)

            pol_chunk, pm_chunk = self._load_chunk(fname, start, chunk_size)

            bs = i * self.chunk_size
            be = bs + chunk_size
            batch_pol[bs:be, 0, :] = pol_chunk["cross"]
            batch_pol[bs:be, 1, :] = pol_chunk["plus"]
            m1_buf[bs:be] = pm_chunk["mass_1"]
            m2_buf[bs:be] = pm_chunk["mass_2"]

        params = _compute_target_params(m1_buf, m2_buf, self.target_parameters)
        return torch.tensor(batch_pol), torch.tensor(params)

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            yield self.sample_batch()


class _ChunkedWaveformParamDataset(torch.utils.data.IterableDataset):
    """Sample batches of (polarizations, params) tuples from a chunk iterator."""

    def __init__(
        self,
        chunk_it: Iterable,
        batch_size: int,
        batches_per_chunk: int,
    ) -> None:
        self.chunk_it = chunk_it
        self.batch_size = batch_size
        self.batches_per_chunk = batches_per_chunk

    def __len__(self) -> int:
        return len(self.chunk_it) * self.batches_per_chunk

    def __iter__(self):
        it = iter(self.chunk_it)
        [pol_chunk], [param_chunk] = next(it)
        num_waveforms = pol_chunk.shape[0]

        while True:
            for _ in range(self.batches_per_chunk):
                idx = torch.randperm(num_waveforms)[: self.batch_size]
                yield pol_chunk[idx], param_chunk[idx]

            try:
                [pol_chunk], [param_chunk] = next(it)
            except StopIteration:
                break
            num_waveforms = pol_chunk.shape[0]


class RegressionTimeDomainDataset(BaseAframeDataset):
    """``BaseAframeDataset`` subclass for GW parameter regression.

    Supports two training waveform modes:
      - **Disk** (``WaveformLoader``): reads pre-generated cross/plus HDF5
        files together with ``mass_1``/``mass_2`` parameter labels.
      - **On-the-fly** (``WaveformGenerator`` / ``CBCGenerator``): samples
        intrinsic parameters from ``training_prior`` each batch, generates
        polarisations on the GPU, and extracts labels from the prior sample.

    Both modes share the same whitening / projection pipeline from
    ``BaseAframeDataset``.

    Training batches: ``(X_whitened, y_params, empty_z)``
    """

    def __init__(
        self,
        *args,
        target_parameters: tuple[str, ...] = ("chirp_mass",),
        n_val_waveforms: int = 4096,
        waveforms_dir: str = ".",
        num_files_per_batch: int = 1,
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
        **kwargs,
    ) -> None:
        # base class requires waveforms_dir/num_files_per_batch; defaults are
        # harmless placeholders when using on-the-fly waveform generation.
        # Defaults must be non-None so Lightning's save_hyperparameters() never
        # captures None, which would crash base.py's get_data_dir call.
        super().__init__(
            *args,
            waveforms_dir=waveforms_dir,
            num_files_per_batch=num_files_per_batch,
            **kwargs,
        )
        self.target_parameters = target_parameters
        self.n_val_waveforms = n_val_waveforms
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def train_val_split(self):
        """Like BaseAframeDataset.train_val_split but also accepts flat dirs
        and skips files shorter than sample_length."""
        import glob as _glob
        from pathlib import Path as _Path

        # Try the standard aframe layout first ({background_dir}/background/*.hdf5)
        fnames = _glob.glob(f"{self.background_dir}/background/*.hdf5")
        if not fnames:
            # Fall back to flat directory (files directly in background_dir)
            fnames = _glob.glob(f"{self.background_dir}/*.hdf5")
        fnames = sorted([_Path(f) for f in fnames])
        durations = [int(f.stem.split("-")[-1]) for f in fnames]

        # Drop files too short to fit even one kernel
        min_dur = self.sample_length  # seconds
        kept = [(f, d) for f, d in zip(fnames, durations) if d >= min_dur]
        n_dropped = len(fnames) - len(kept)
        if n_dropped:
            self._logger.warning(
                f"Dropped {n_dropped} files shorter than sample_length={min_dur:.1f}s"
            )
        fnames, durations = zip(*kept) if kept else ([], [])
        fnames, durations = list(fnames), list(durations)

        valid_fnames, valid_duration = [], 0
        while valid_duration < self.hparams.min_valid_duration:
            fname, duration = fnames.pop(-1), durations.pop(-1)
            valid_duration += duration
            valid_fnames.append(str(fname))
        train_fnames = [str(f) for f in fnames]
        train_duration = sum(durations)
        self._logger.info(
            f"Using {len(train_fnames)} files ({train_duration / 86400:.2f} days) for training"
        )
        self._logger.info(
            f"Using {len(valid_fnames)} files ({valid_duration / 86400:.2f} days) for validation"
        )
        return train_fnames, valid_fnames

    def slice_waveforms(self, waveforms: torch.Tensor) -> torch.Tensor:
        # If the waveform sampler has a window_offset attribute (e.g. CBCGenerator),
        # shift signal_idx backwards by that many seconds so the model window ends
        # window_offset seconds before physical coalescence.
        # This avoids touching base.py.
        window_offset = getattr(self.waveform_sampler, "window_offset", 0.0)
        if window_offset == 0.0:
            return super().slice_waveforms(waveforms)
        sr = self.hparams.sample_rate
        signal_idx = waveforms.shape[-1] - int(
            (self.waveform_sampler.right_pad + window_offset) * sr
        )
        kernel_size = int(self.hparams.kernel_length * sr) + self.filter_size
        if kernel_size < self.left_pad_size + self.right_pad_size:
            raise ValueError(
                f"Kernel size ({kernel_size}) cannot be less than total "
                f"padding ({self.left_pad_size} + {self.right_pad_size})"
            )
        start_idx = signal_idx - (kernel_size - self.right_pad_size)
        stop_idx = signal_idx + (kernel_size - self.left_pad_size)
        left_pad = -1 * min(start_idx, 0)
        right_pad = max(stop_idx - waveforms.shape[-1], 0)
        if left_pad > 0:
            start_idx += left_pad
            stop_idx += left_pad
        waveforms = torch.nn.functional.pad(waveforms, [left_pad, right_pad])
        return waveforms[..., start_idx:stop_idx]

    def setup(self, stage: str) -> None:
        world_size, rank = self.get_world_size_and_rank()
        self._logger = self.get_logger(world_size, rank)
        self.train_fnames, self.valid_fnames = self.train_val_split()

        with h5py.File(self.train_fnames[0], "r") as f:
            self._raw_sample_rate = int(round(1 / f[self.hparams.ifos[0]].attrs["dx"]))

        target_rate = int(self.hparams.sample_rate)
        if self._raw_sample_rate != target_rate:
            import torchaudio
            self.resampler = torchaudio.transforms.Resample(
                self._raw_sample_rate, target_rate
            )
        else:
            self.resampler = None

        self.build_transforms()
        self.transforms_to_device()

        # Sample fixed validation prior parameters with a fixed seed so
        # validation waveforms are consistent across epochs.
        rng_state = torch.get_rng_state()
        torch.manual_seed(42 + rank)
        n = self.n_val_waveforms // world_size
        self._val_prior_params = self.waveform_sampler.training_prior(n, device="cpu")
        self.val_params = _compute_target_params_tensor(
            self._val_prior_params, self.target_parameters
        )
        torch.set_rng_state(rng_state)

    # ------------------------------------------------------------------ #
    # Batch transfer hooks                                                 #
    # ------------------------------------------------------------------ #

    def on_before_batch_transfer(self, batch, _):
        if self.trainer.training and self.waveforms_from_disk:
            # Unpack (pol, params) tuple before device transfer; slice pol.
            X, (polarizations, params) = batch
            polarizations = self.slice_waveforms(polarizations)
            batch = X, (polarizations, params)
        return batch

    def on_after_batch_transfer(self, batch, _):
        if self.trainer.training:
            if self.waveforms_from_disk:
                [X], (waveforms, params) = batch
                batch = self.inject(X, waveforms, params)
            else:
                [X] = batch
                batch = self._inject_from_generator(X)
        elif self.trainer.validating or self.trainer.sanity_checking:
            [X] = batch
            batch = self._inject_fixed_val(X)
        return batch

    # ------------------------------------------------------------------ #
    # Injection                                                            #
    # ------------------------------------------------------------------ #

    def _project_inject_whiten(self, X, waveforms, psds, params):
        """Shared projection + injection + whitening; returns regression batch."""
        B = X.shape[0]
        dec, psi, phi = self.sample_extrinsic(X)
        snrs = (
            self.snr_sampler.sample((B,)).to(X.device)
            if self.snr_sampler is not None
            else None
        )
        responses = self.projector(
            dec, psi, phi, snrs, psds,
            cross=waveforms[:, 0], plus=waveforms[:, 1],
        )
        kernels = sample_kernels(responses, kernel_size=X.size(-1), coincident=True)
        X = self.whitener(X + kernels, psds)
        empty_z = torch.empty(B, 0, device=X.device)
        return X, params, empty_z

    def _resample(self, X: torch.Tensor) -> torch.Tensor:
        if self.resampler is not None:
            X = self.resampler(X)
        return X

    @torch.no_grad()
    def inject(self, X, waveforms, params):
        """Disk path: waveforms and params already loaded; inject all samples."""
        X = self._resample(X)
        X, psds = self.psd_estimator(X)
        X = self.inverter(X)
        X = self.reverser(X)

        B = X.shape[0]
        idx = torch.randperm(waveforms.shape[0])[:B]
        waveforms = waveforms[idx].to(X.device).float()
        params = params[idx].to(X.device).float()
        return self._project_inject_whiten(X, waveforms, psds, params)

    @torch.no_grad()
    def _inject_from_generator(self, X):
        """On-the-fly path: sample prior, generate waveforms, extract params."""
        X = self._resample(X)
        X, psds = self.psd_estimator(X)
        X = self.inverter(X)
        X = self.reverser(X)

        B = X.shape[0]
        prior_params = self.waveform_sampler.training_prior(B, device=X.device)
        hc, hp = self.waveform_sampler(**prior_params)
        waveforms = torch.stack([hc, hp], dim=1).float()  # (B, 2, L_wf)

        # Slice to kernel length (generator produces full-duration waveforms)
        waveforms = self.slice_waveforms(waveforms)
        params = _compute_target_params_tensor(prior_params, self.target_parameters)
        return self._project_inject_whiten(X, waveforms, psds, params)

    @torch.no_grad()
    def _inject_fixed_val(self, X):
        """Validation: inject fixed waveforms (reproducible across epochs)."""
        X = self._resample(X)
        X, psds = self.psd_estimator(X)
        # No random augmentation (inverter/reverser) during validation

        B = X.shape[0]
        idx = torch.randperm(len(self.val_params))[:B]
        prior_params = {
            k: v[idx].to(X.device) for k, v in self._val_prior_params.items()
        }
        hc, hp = self.waveform_sampler(**prior_params)
        waveforms = torch.stack([hc, hp], dim=1).float()
        waveforms = self.slice_waveforms(waveforms)
        params = self.val_params[idx].to(X.device)
        return self._project_inject_whiten(X, waveforms, psds, params)

    # ------------------------------------------------------------------ #
    # Dataloaders                                                          #
    # ------------------------------------------------------------------ #

    def train_dataloader(self) -> ZippedDataset:
        from ml4gw.dataloading import Hdf5TimeSeriesDataset

        # kernel_size uses raw (pre-resample) rate; resampling happens per batch
        raw_kernel = int(self._raw_sample_rate * self.sample_length)
        bg_dataset = Hdf5TimeSeriesDataset(
            self.train_fnames,
            channels=self.hparams.ifos,
            kernel_size=raw_kernel,
            batch_size=self.hparams.batch_size,
            batches_per_epoch=self.batches_per_epoch,
            coincident=False,
            num_files_per_batch=self.hparams.num_files_per_batch,
        )
        pin_memory = isinstance(
            self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        )
        bg_loader = torch.utils.data.DataLoader(
            bg_dataset,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )

        if not self.waveforms_from_disk:
            # Generator path: background only; waveforms produced in inject().
            return bg_loader

        waveform_loader = _WaveformParamLoader(
            fnames=self.waveform_sampler.training_waveform_files,
            target_parameters=self.target_parameters,
            batch_size=self.hparams.chunk_size,
            batches_per_epoch=self.hparams.chunks_per_epoch or 1,
        )

        world_size, _ = self.get_world_size_and_rank()
        batches_per_epoch = self.hparams.batches_per_epoch // world_size
        batches_per_chunk = int(batches_per_epoch // self.hparams.chunks_per_epoch) + 1

        wf_dl = torch.utils.data.DataLoader(
            waveform_loader,
            num_workers=2,
            pin_memory=pin_memory,
            persistent_workers=True,
        )
        wf_dataset = _ChunkedWaveformParamDataset(
            wf_dl,
            batch_size=self.hparams.batch_size,
            batches_per_chunk=batches_per_chunk,
        )
        return ZippedDataset(bg_loader, wf_dataset)

    def val_dataloader(self):
        from ml4gw.dataloading import Hdf5TimeSeriesDataset

        val_batches = max(1, self.batches_per_epoch // 10)
        val_dataset = Hdf5TimeSeriesDataset(
            self.valid_fnames,
            channels=self.hparams.ifos,
            kernel_size=int(self._raw_sample_rate * self.sample_length),
            batch_size=self.hparams.batch_size,
            batches_per_epoch=val_batches,
            coincident=False,
            num_files_per_batch=max(1, len(self.valid_fnames) // 4),
        )
        pin_memory = isinstance(
            self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )
