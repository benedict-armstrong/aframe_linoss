"""Generate an InterferometerResponseSet for BNS sensitive-volume evaluation.

Produces waveforms.hdf5 + rejected-parameters.hdf5 in the output directory,
covering the full O3a GPS range at 4096 Hz so they can be injected into
the O3a background during regression_infer.py.

Usage
-----
    cd projects/train
    uv run python generate_bns_injection_set.py --output_dir /path/to/injections/
"""

import argparse
import logging
import os
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def get_gps_range(background_dir: str) -> tuple[float, float]:
    files = sorted(Path(background_dir).glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files in {background_dir}")
    starts, ends = [], []
    for f in files:
        with h5py.File(f, "r") as hf:
            x0 = float(hf["H1"].attrs["x0"])
            dx = float(hf["H1"].attrs["dx"])
            n = len(hf["H1"])
            starts.append(x0)
            ends.append(x0 + n * dx)
    return min(starts), max(ends)


def main(
    background_dir: str,
    output_dir: str,
    spacing: float = 64.0,
    buffer: float = 64.0,
    snr_threshold: float = 0.0,
    sample_rate: float = 4096.0,
    waveform_duration: float = 64.0,
    seed: int = 42,
):
    from priors.priors import end_o3_ratesandpops_bns
    from data.waveforms.testing import testing_waveforms

    start, end = get_gps_range(background_dir)
    log.info(f"O3a GPS range: {start:.0f} – {end:.0f}  ({(end-start)/3600:.1f} hr)")

    # Use first background file for PSD estimation
    psd_file = sorted(Path(background_dir).glob("*.hdf5"))[0]
    log.info(f"PSD file: {psd_file}")

    waveform_fname, rejected_fname = testing_waveforms(
        start=start,
        end=end,
        ifos=["H1", "L1"],
        shifts=[0.0, 0.0],
        spacing=spacing,
        buffer=buffer,
        prior=end_o3_ratesandpops_bns,
        minimum_frequency=20.0,
        reference_frequency=50.0,
        sample_rate=sample_rate,
        waveform_duration=waveform_duration,
        waveform_approximant="IMRPhenomPv2",
        right_pad=0.0,
        highpass=20.0,
        lowpass=None,
        snr_threshold=snr_threshold,
        psd_file=psd_file,
        max_num_samples=4096,
        output_dir=Path(output_dir),
        seed=seed,
    )

    log.info(f"Injection set written to: {waveform_fname}")
    log.info(f"Rejected params written to: {rejected_fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--spacing", type=float, default=64.0, help="Seconds between injections")
    parser.add_argument("--buffer", type=float, default=64.0, help="Edge buffer in seconds")
    parser.add_argument("--snr_threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(**vars(args))
