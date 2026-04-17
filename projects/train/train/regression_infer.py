"""Offline regression inference pipeline.

Produces the same ``background.hdf5`` (``EventSet``) and
``foreground.hdf5`` (``RecoveredInjectionSet``) files that the standard
aframe Triton pipeline produces, so the existing ``plots/legacy/main.py``
sensitive-volume calculation works unchanged.

Detection statistic: ``sigma_chirp_mass = sqrt(softplus(var[:, 0]))``

Usage
-----
    uv run python -m train.regression_infer \\
        --config regression_infer.yaml

Minimal YAML
------------
    checkpoint: path/to/checkpoint.ckpt
    model_class: LitLinOSSGaussianNLL        # or LitS4DGaussianNLL
    background_dir: /data/background
    injection_set_fname: /data/injections.hdf5
    ifos: [H1, L1]
    shifts: [[0, 1], [0, 2], [0, 3]]         # seconds per IFO
    sample_rate: 2048
    kernel_length: 1.0
    fduration: 0.5
    psd_length: 64.0
    fftlength: 2.0
    inference_sampling_rate: 8.0
    integration_window_length: 1.0
    cluster_window_length: 0.5
    highpass: 32.0
    batch_size: 128
    outdir: /results/regression_sv
"""

import glob
import logging
import math
from pathlib import Path
from typing import Optional

import h5py
import jsonargparse
import numpy as np
import torch
import torch.nn as nn
from ml4gw.transforms import Whiten
from tqdm import tqdm

from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InterferometerResponseSet, waveform_class_factory
from utils.preprocessing import PsdEstimator

log = logging.getLogger(__name__)

SECONDS_PER_YEAR = 31_556_952


# ────────────────────────────────────────────────────────────────────────── #
# Data iterator                                                               #
# ────────────────────────────────────────────────────────────────────────────#

class RegressionSequence:
    """Windowed data iterator for direct (non-Triton) regression inference.

    Mirrors ``infer.data.Sequence`` but yields batches of fixed-length windows
    suitable for a windowed (non-streaming) model like S4D or LinOSS.

    For background timeslides, different IFOs are shifted in time relative to
    each other so that coincident noise triggers are de-correlated.  For
    foreground (injections), all shifts are zero.

    Args:
        background_fname:
            Path to an HDF5 background segment.  Each IFO channel is stored
            as a dataset keyed by IFO name (e.g. ``H1``) with attributes
            ``dx`` (1/sample_rate) and ``x0`` (GPS start time).
        injection_set_fname:
            Path to an ``InterferometerResponseSet`` HDF5 file.
        ifos:
            List of IFO names matching the background file.
        shifts:
            Time shift in *seconds* to apply to each IFO.
            ``[0.0, 1.0]`` means IFO-0 is the reference, IFO-1 is shifted
            forward by 1 s (so they look at non-coincident background).
        sample_length:
            Length in seconds of each window fed to the model:
            ``kernel_length + fduration + psd_length``.
        inference_sampling_rate:
            Rate at which inference outputs are produced (windows/s).
        batch_size:
            Number of windows per forward pass.
    """

    def __init__(
        self,
        background_fname: str,
        injection_set_fname: str,
        ifos: list[str],
        shifts: list[float],
        sample_length: float,
        inference_sampling_rate: float,
        batch_size: int,
    ) -> None:
        self.background_fname = background_fname
        self.ifos = ifos
        self.batch_size = batch_size
        self.inference_sampling_rate = inference_sampling_rate

        with h5py.File(background_fname, "r") as f:
            ds = f[ifos[0]]
            self.sample_rate: float = 1.0 / ds.attrs["dx"]
            self.t0: float = ds.attrs["x0"]
            self.size: int = len(ds)

        self.duration = self.size / self.sample_rate
        self.sample_length_samples = int(sample_length * self.sample_rate)
        self.stride = int(self.sample_rate / inference_sampling_rate)
        # shifts in samples
        self.shifts_samples = [int(s * self.sample_rate) for s in shifts]

        # Load injections that fall in this segment (zero-lag, all shifts)
        cls = waveform_class_factory(
            ifos, InterferometerResponseSet, "ResponseSet"
        )
        inj = cls.read(
            injection_set_fname,
            start=self.t0,
            end=self.t0 + self.duration,
            shifts=[0.0] * len(ifos),
        )
        self.injection_set = inj if len(inj) > 0 else None
        if self.injection_set is None:
            log.info(
                f"No injections in {background_fname} for zero-lag — "
                "foreground inference will be skipped."
            )

    # ------------------------------------------------------------------ #

    @property
    def max_shift(self) -> int:
        return max(self.shifts_samples)

    @property
    def n_steps(self) -> int:
        """Number of output steps (length of score timeseries)."""
        usable = self.size - self.max_shift - self.sample_length_samples
        return max(0, usable // self.stride)

    def __len__(self) -> int:
        return math.ceil(self.n_steps / self.batch_size)

    def _load_shifted(self) -> np.ndarray:
        """Return background with time shifts applied.

        After shifting, all IFOs are aligned in the same index space: index p
        corresponds to GPS time ``t0 + p / sample_rate`` for the reference IFO.
        """
        with h5py.File(self.background_fname, "r") as f:
            arrays = [
                f[ifo][sh : self.size - (self.max_shift - sh)]
                for ifo, sh in zip(self.ifos, self.shifts_samples, strict=True)
            ]
        return np.stack(arrays).astype(np.float32)  # (n_ifos, n_valid)

    def _load_unshifted(self) -> np.ndarray:
        """Return background with no shifts (for foreground injection)."""
        with h5py.File(self.background_fname, "r") as f:
            arrays = [f[ifo][:] for ifo in self.ifos]
        return np.stack(arrays).astype(np.float32)  # (n_ifos, size)

    def __iter__(self):
        """Yield ``(x_bg, x_inj)`` batches.

        Both arrays have shape ``(B, n_ifos, sample_length_samples)`` where
        ``B <= batch_size``.  ``x_inj`` is ``None`` if there are no injections.
        """
        bg = self._load_shifted()     # (n_ifos, n_valid)

        # Build foreground once up front for the whole segment.
        fg = None
        if self.injection_set is not None:
            unshifted = self._load_unshifted()
            # inject() takes (n_ifos, N) and GPS start time
            injected = self.injection_set.inject(unshifted.copy(), self.t0)
            # Trim to the same valid length as bg (max-shift window)
            fg = injected[:, : bg.shape[1]]

        W = self.sample_length_samples
        n = self.n_steps

        for b0 in range(0, n, self.batch_size):
            b1 = min(b0 + self.batch_size, n)
            B = b1 - b0

            x_bg = np.empty((B, len(self.ifos), W), dtype=np.float32)
            for k, step in enumerate(range(b0, b1)):
                s = step * self.stride
                x_bg[k] = bg[:, s : s + W]

            x_inj = None
            if fg is not None:
                x_inj = np.empty_like(x_bg)
                for k, step in enumerate(range(b0, b1)):
                    s = step * self.stride
                    x_inj[k] = fg[:, s : s + W]

            yield x_bg, x_inj

    def recover(self, fg_events: EventSet) -> RecoveredInjectionSet:
        """Match foreground events to injections by time."""
        return RecoveredInjectionSet.recover(fg_events, self.injection_set)


# ────────────────────────────────────────────────────────────────────────── #
# Postprocessing (replaces infer.postprocess.Postprocessor)                   #
# ────────────────────────────────────────────────────────────────────────────#

def _integrate(y: np.ndarray, window_size: int) -> np.ndarray:
    """Boxcar integration matching ``Postprocessor.integrate``."""
    window = np.ones(window_size) / window_size
    integrated = np.convolve(y, window, mode="full")
    return integrated[: -window_size + 1]


def _cluster(
    y: np.ndarray,
    t0: float,
    shifts: list[float],
    inference_sampling_rate: float,
    cluster_window_size: int,
    psd_offset: int,
) -> EventSet:
    """Sliding-window local-max clustering matching ``Postprocessor.cluster``."""
    y = y[psd_offset:]
    half = cluster_window_size // 2
    i = int(np.argmax(y[:half]))

    events, times = [], []
    while i < len(y):
        val = y[i]
        window = y[i + 1 : i + 1 + half]
        if len(window) and (val < window).any():
            i += int(np.argmax(window)) + 1
        else:
            events.append(val)
            times.append(t0 + i / inference_sampling_rate)
            i += half + 1

    Tb = len(y) / inference_sampling_rate
    ev = np.array(events, dtype=np.float64)
    tm = np.array(times, dtype=np.float64)
    sh = np.tile(shifts, (len(ev), 1)).astype(np.float64)
    return EventSet(ev, tm, sh, Tb)


def _postprocess(
    y: np.ndarray,
    t0: float,
    shifts: list[float],
    psd_length: float,
    fduration: float,
    inference_sampling_rate: float,
    integration_window_length: float,
    cluster_window_length: float,
) -> EventSet:
    """Full integrate → cluster pipeline."""
    isr = inference_sampling_rate
    integration_window_size = int(isr * integration_window_length) + 1
    cluster_window_size = int(isr * cluster_window_length)
    psd_offset = int(psd_length * isr)

    y = _integrate(y, integration_window_size)
    # Adjust t0 for the data that got consumed during integration
    t0_out = t0 + psd_length - fduration / 2 - integration_window_length
    return _cluster(y, t0_out, shifts, isr, cluster_window_size, psd_offset)


# ────────────────────────────────────────────────────────────────────────── #
# Inference                                                                   #
# ────────────────────────────────────────────────────────────────────────────#

@torch.no_grad()
def score_sequence(
    model: nn.Module,
    sequence: RegressionSequence,
    psd_estimator: PsdEstimator,
    whitener: Whiten,
    device: torch.device,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Run the regression model over one background segment.

    Returns:
        bg_scores : (T,) float32 — ``sigma_chirp_mass`` for background
        fg_scores : (T,) float32 or None — same for injected foreground
    """
    n_vars = model.n_vars
    softplus = nn.Softplus()

    def _score(x_np: np.ndarray) -> np.ndarray:
        # x_np: (B, n_ifos, sample_length_samples)
        x = torch.from_numpy(x_np).to(device)
        x, psds = psd_estimator(x)          # (B, n_ifos, L_kernel)
        x = whitener(x, psds)               # (B, n_ifos, L_whiten)
        x = model._prepare_input(x)         # transpose if LinOSS
        out = model(x)                      # (B, 2*n_vars)
        var = softplus(out[:, n_vars:])     # (B, n_vars)
        sigma = torch.sqrt(var[:, 0])       # (B,)  chirp_mass sigma
        return sigma.cpu().numpy()

    bg_parts, fg_parts = [], []
    has_fg = False

    for x_bg, x_inj in tqdm(sequence, total=len(sequence), leave=False):
        bg_parts.append(_score(x_bg))
        if x_inj is not None:
            fg_parts.append(_score(x_inj))
            has_fg = True
        elif has_fg:
            # pad with bg scores to keep timeseries aligned
            fg_parts.append(bg_parts[-1])

    bg_scores = np.concatenate(bg_parts) if bg_parts else np.array([])
    fg_scores = np.concatenate(fg_parts) if has_fg else None
    return bg_scores, fg_scores


# ────────────────────────────────────────────────────────────────────────── #
# Main                                                                        #
# ────────────────────────────────────────────────────────────────────────────#

def main(
    checkpoint: str,
    model_class: str,
    background_dir: str,
    injection_set_fname: str,
    ifos: list[str],
    shifts: list[list[float]],
    sample_rate: float,
    kernel_length: float,
    fduration: float,
    psd_length: float,
    fftlength: float,
    inference_sampling_rate: float,
    integration_window_length: float,
    cluster_window_length: float,
    outdir: str,
    highpass: Optional[float] = None,
    lowpass: Optional[float] = None,
    batch_size: int = 128,
    device: str = "cuda",
    verbose: bool = False,
) -> None:
    """Aggregate inference over all background segments and all shift combinations.

    Writes:
        ``{outdir}/background.hdf5`` — ``EventSet``
        ``{outdir}/foreground.hdf5`` — ``RecoveredInjectionSet``
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────── #
    log.info(f"Loading {model_class} from {checkpoint}")
    from train.model.regression import LitS4DGaussianNLL, LitLinOSSGaussianNLL
    cls_map = {
        "LitS4DGaussianNLL": LitS4DGaussianNLL,
        "LitLinOSSGaussianNLL": LitLinOSSGaussianNLL,
    }
    if model_class not in cls_map:
        raise ValueError(
            f"Unknown model_class {model_class!r}. Choose from {list(cls_map)}"
        )
    model = cls_map[model_class].load_from_checkpoint(checkpoint)
    model.eval()

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    log.info(f"Model loaded, running on {dev}")

    # ── build preprocessing modules ──────────────────────────────────── #
    window_length = kernel_length + fduration
    psd_estimator = PsdEstimator(
        window_length,
        sample_rate,
        fftlength,
        average="median",
        fast=(highpass is not None),
    ).to(dev)
    whitener = Whiten(fduration, sample_rate, highpass, lowpass).to(dev)

    sample_length = kernel_length + fduration + psd_length

    # ── find background files ─────────────────────────────────────────── #
    bg_files = sorted(glob.glob(str(Path(background_dir) / "*.hdf5")))
    if not bg_files:
        raise FileNotFoundError(f"No HDF5 files found in {background_dir}")
    log.info(f"Found {len(bg_files)} background segments, {len(shifts)} shifts")

    # ── run inference ─────────────────────────────────────────────────── #
    all_bg = EventSet()
    all_fg = RecoveredInjectionSet()

    for fname in bg_files:
        for shift_combo in shifts:
            log.info(
                f"  segment {Path(fname).name}  shifts={shift_combo}"
            )
            seq = RegressionSequence(
                background_fname=fname,
                injection_set_fname=injection_set_fname,
                ifos=ifos,
                shifts=shift_combo,
                sample_length=sample_length,
                inference_sampling_rate=inference_sampling_rate,
                batch_size=batch_size,
            )

            if seq.n_steps == 0:
                log.warning(f"  Segment too short, skipping.")
                continue

            bg_ts, fg_ts = score_sequence(
                model, seq, psd_estimator, whitener, dev
            )

            # background events
            bg_events = _postprocess(
                bg_ts,
                t0=seq.t0,
                shifts=shift_combo,
                psd_length=psd_length,
                fduration=fduration,
                inference_sampling_rate=inference_sampling_rate,
                integration_window_length=integration_window_length,
                cluster_window_length=cluster_window_length,
            )
            all_bg = all_bg.append(bg_events)

            # foreground events (zero-lag only)
            if fg_ts is not None and seq.injection_set is not None:
                fg_events = _postprocess(
                    fg_ts,
                    t0=seq.t0,
                    shifts=[0.0] * len(ifos),
                    psd_length=psd_length,
                    fduration=fduration,
                    inference_sampling_rate=inference_sampling_rate,
                    integration_window_length=integration_window_length,
                    cluster_window_length=cluster_window_length,
                )
                recovered = seq.recover(fg_events)
                all_fg = all_fg.append(recovered)

    # ── write outputs ─────────────────────────────────────────────────── #
    bg_path = outdir / "background.hdf5"
    fg_path = outdir / "foreground.hdf5"
    all_bg.write(bg_path)
    all_fg.write(fg_path)
    log.info(
        f"Done.  background events: {len(all_bg)}  "
        f"(Tb={all_bg.Tb/SECONDS_PER_YEAR:.2f} yr)  |  "
        f"foreground events: {len(all_fg)}"
    )
    log.info(f"Results written to {outdir}")


def cli():
    parser = jsonargparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action=jsonargparse.ActionConfigFile)
    parser.add_function_arguments(main)
    cfg = parser.parse_args()
    cfg = parser.instantiate_classes(cfg)
    main(**cfg)


if __name__ == "__main__":
    cli()
