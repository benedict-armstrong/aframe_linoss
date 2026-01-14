import logging
import math
import time
from contextlib import nullcontext
from typing import Optional
from zlib import adler32

import h5py
import numpy as np
import requests

from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InterferometerResponseSet, waveform_class_factory


class MetricsRateLimiter:
    """
    Rate limiter that polls Triton server metrics and waits
    when pending requests exceed a threshold.
    """

    def __init__(
        self,
        metrics_url: str,
        max_pending_requests: int = 5000,
        poll_interval: float = 0.1,
        model: str = "preprocessor",
    ):
        """
        Args:
            metrics_url:
                URL to the Triton metrics endpoint (e.g., "http://g145.internal.cluster.is.localnet:8002/metrics")
            max_pending_requests:
                Maximum number of pending requests before throttling
            poll_interval:
                How often to poll metrics when waiting (in seconds)
            model:
                The model name to check pending requests for
        """
        self.metrics_url = metrics_url
        self.max_pending_requests = max_pending_requests
        self.poll_interval = poll_interval
        self.model = model

    def _get_pending_requests(self) -> int:
        """Fetch the current pending request count from Triton metrics."""
        try:
            response = requests.get(self.metrics_url, timeout=2)
            response.raise_for_status()
            for line in response.text.splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                # Look for nv_inference_pending_request_count for our model
                if (
                    "nv_inference_pending_request_count" in line
                    and f'model="{self.model}"' in line
                ):
                    try:
                        value = float(line.rsplit(" ", 1)[1])
                        return int(value)
                    except (IndexError, ValueError):
                        continue
            return 0
        except requests.RequestException as e:
            logging.warning(f"Failed to fetch metrics: {e}")
            # If we can't fetch metrics, don't block
            return 0

    def wait_if_needed(self):
        """Wait until pending requests drop below the threshold."""
        pending = self._get_pending_requests()
        if pending >= self.max_pending_requests:
            logging.debug(
                f"Pending requests ({pending}) >= threshold "
                f"({self.max_pending_requests}), waiting..."
            )
        while pending >= self.max_pending_requests:
            time.sleep(self.poll_interval)
            pending = self._get_pending_requests()

    def __enter__(self):
        self.wait_if_needed()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def build_metrics_url(address: str) -> str:
    """
    Build the Triton metrics URL from the inference server address.

    Args:
        address: Triton server address (e.g., "g145:8001" or
                 "g145.internal.cluster.is.localnet:8001")

    Returns:
        Metrics URL (e.g., "http://g145.internal.cluster.is.localnet:8002/metrics")
    """
    # Remove any protocol prefix if present
    if "://" in address:
        address = address.split("://", 1)[1]

    # Split host and port
    if ":" in address:
        host = address.rsplit(":", 1)[0]
    else:
        host = address

    print(f"Built metrics URL for Triton server at host: {host}")

    return f"http://{host}:8002/metrics"


class Sequence:
    def __init__(
        self,
        background_fname: str,
        injection_set_fname: str,
        ifos: list[str],
        shifts: list[float],
        inference_sampling_rate: float,
        batch_size: int,
        triton_address: str,
        max_pending_requests: int = 5000,
        **kwargs,
    ):
        """
        Object used for iterating over a segment of data,
        performing timeshifts, optionally injecting waveforms, and
        aggregating the returned inference outputs.

        If the injection set is empty for this given
        segment and shifts, inference on injections will be skipped,
        and `None` will be returned for the foreground events.

        Args:
            background_fname:
                Path to the background segment
            injection_set_fname:
                Path to the injection set file
            ifos:
                Interferometer names
            shifts:
                Time shifts to apply to each interferometer
            inference_sampling_rate:
                Rate at which inference is performed
            batch_size:
                Number of inference requests to send to the model at once
            triton_address:
                Address of the Triton inference server (e.g., "g145.internal.cluster.is.localnet").
                Used to poll metrics for rate limiting. If None, no rate limiting.
            max_pending_requests:
                Maximum pending requests before throttling (default 5000)
        """
        logging.info("Initializing sequence")

        self.background_fname = background_fname
        self.inference_sampling_rate = inference_sampling_rate
        self.batch_size = batch_size
        self.metrics_url = build_metrics_url(triton_address)
        self.max_pending_requests = max_pending_requests
        self.ifos = ifos

        print(
            f"Initializing sequence with background file {background_fname}"
            f" and injection set file {injection_set_fname}"
            f" for ifos {ifos} and shifts {shifts}"
            f" at inference sampling rate {inference_sampling_rate}"
            f" and batch size {batch_size}"
        )

        if len(ifos) != len(shifts):
            raise ValueError(
                "Number of ifos must match number of shifts"
                f"got {len(ifos)} ifos and {len(shifts)} shifts"
            )

        # read some of the metadata from our background file
        with h5py.File(background_fname, "r") as f:
            dataset = f[ifos[0]]
            self.size = len(dataset)
            self.sample_rate = 1 / dataset.attrs["dx"]
            self.t0 = dataset.attrs["x0"]
            self.duration = self.size / self.sample_rate

        # load in our injections up front
        # if there are no injections for
        # this shift, set it to None so
        # we don't run inference on injections

        cls = waveform_class_factory(
            ifos,
            InterferometerResponseSet,
            "ResponseSet",
        )

        injection_set = cls.read(
            injection_set_fname,
            start=self.t0,
            end=self.t0 + self.duration,
            shifts=shifts,
        )
        if len(injection_set) == 0:
            logging.info(
                f"No injections found in {injection_set_fname} "
                f"for segment {background_fname} and "
                f"shifts {shifts}, skipping."
            )
            injection_set = None

        self.injection_set = injection_set

        # derive some properties from that metadata,
        # including come up with a semi-unique sequence
        # id derived from a hash of its most descriptive parts
        fingerprint = f"{self.t0}{self.duration}{shifts}".encode()
        self.id = adler32(fingerprint)
        self.shifts = [int(i * self.sample_rate) for i in shifts]
        self.stride = int(self.sample_rate / inference_sampling_rate)
        self.step_size = self.stride * batch_size
        # initialize some containers for handling during
        # the inference response callback
        self._started = {}
        self._done = {}
        self._sequences = {}
        size = len(self) * self.batch_size
        for i in range(2):
            seq_id = self.id + i
            self._started[seq_id] = False
            self._done[seq_id] = False
            self._sequences[seq_id] = np.zeros(size)

        # if there are no injections, we can mark
        # the injection sequence as started and done
        if self.injection_set is None:
            self._done[self.id + 1] = True
            self._started[self.id + 1] = True

    @property
    def started(self):
        return all(self._started.values())

    @property
    def done(self):
        return all(self._done.values())

    @property
    def remainder(self):
        # the number of remaining data points not filling a full batch
        return (self.size - max(self.shifts)) % self.step_size

    @property
    def num_pad(self):
        # the number of zeros we need to pad the last batch
        # to make it a full batch
        return (self.step_size - self.remainder) % self.step_size

    @property
    def slice(self) -> slice:
        """
        The number of inference requests we need to slice
        off the end of the sequences to remove
        the dummy data from the last batch
        """

        # if num_pad is 0 don't slice anything
        num_slice = self.num_pad // self.stride
        end = -num_slice if num_slice else None
        return slice(end)

    def __len__(self):
        # this include excess data at end of sequence that can't
        # be used for a full batch. We'll end up padding it
        # with zeros to make it a full batch and
        # slicing off the actual useful inference requests
        # corresponding to the excess
        return math.ceil((self.size - max(self.shifts)) / self.step_size)

    def __iter__(self):
        if self.metrics_url is not None:
            limiter = MetricsRateLimiter(
                metrics_url=self.metrics_url,
                max_pending_requests=self.max_pending_requests,
            )
        else:
            limiter = nullcontext()

        with h5py.File(self.background_fname, "r") as f:
            for i in range(len(self)):
                # if this is the last batch, we may need to pad it
                # to make it a full batch
                last = i == len(self) - 1
                # grab the current batch of updates from the file
                # and stack it into a 2D array
                x = []
                for ifo, shift in zip(self.ifos, self.shifts):
                    start = shift + i * self.step_size

                    # for all but last batch just
                    # increase by step size
                    end = start + self.step_size

                    # if this is the last batch
                    # and we need to pad it
                    # just step by the remainder
                    if last and self.remainder:
                        end = start + self.remainder

                    data = f[ifo][start:end]
                    # if this is the last batch
                    # possibly pad it to make it a full batch
                    if last:
                        data = np.pad(data, (0, self.num_pad), "constant")

                    x.append(data)
                x = np.stack(x).astype(np.float32)
                # if there are any injections for this shift,
                # inject waveforms into a copy of the background
                x_inj = None
                offset = i * self.batch_size / self.inference_sampling_rate
                if self.injection_set is not None:
                    x_inj = self.injection_set.inject(
                        x.copy(), self.t0 + offset
                    )

                # return the two sets of updates, possibly
                # rate limited if we specified a max rate
                with limiter:
                    yield x, x_inj

    def __call__(self, y, request_id, sequence_id):
        # insert the response at the appropriate
        # spot in the corresponding output array
        start = request_id * self.batch_size
        stop = (request_id + 1) * self.batch_size
        self._sequences[sequence_id][start:stop] = y[:, 0]

        # indicate that the first response for
        # this sequence has returned, and possibly
        # that the last one has returned as well
        self._started[sequence_id] = True
        if request_id == len(self) - 1:
            self._done[sequence_id] = True

        # if both the background and foreground
        # sequences have completed, return them both,
        # slicing off the dummy data from the last batch
        if self.done:
            background = self._sequences[self.id][self.slice]
            foreground = None
            if self.injection_set is not None:
                foreground = self._sequences[self.id + 1][self.slice]
            return background, foreground

    def recover(self, foreground: EventSet) -> RecoveredInjectionSet:
        return RecoveredInjectionSet.recover(foreground, self.injection_set)
