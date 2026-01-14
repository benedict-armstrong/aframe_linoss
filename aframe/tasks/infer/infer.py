import json
import logging
import os
import socket
import time
from contextlib import ExitStack
from pathlib import Path

import h5py
import law
import luigi
import numpy as np
import psutil
from hermes.aeriel.monitor import ServerMonitor
from hermes.aeriel.serve import serve
from luigi.util import inherits

from aframe.base import AframeSingularityTask
from aframe.config import paths
from aframe.parameters import PathParameter
from aframe.tasks.infer.base import InferBase, InferParameters
from aframe.tasks.infer.triton import (
    TritonServerPool,
    TritonServerTask,
)


@inherits(InferParameters)
class DeployInferLocal(InferBase):
    """
    Launch inference on local gpus
    """

    triton_image = luigi.Parameter()

    @staticmethod
    def get_ip_address() -> str:
        """
        Get the local nodes cluster-internal IP address
        """
        # for _, addrs in psutil.net_if_addrs().items():
        #     for addr in addrs:
        #         if (
        #             addr.family == socket.AF_INET
        #             and not addr.address.startswith("127.")
        #         ):
        #             return addr.address
        # raise ValueError("No valid IP address found")
        hostname = socket.gethostname() + ".internal.cluster.is.localnet"
        print(f"Hostname: {hostname}")
        return hostname

    @property
    def model_repo_dir(self):
        # return self.input()["model_repository"].path
        return "/home/barmstrong/aframe_new/runs/model_repository"

    def htcondor_workflow_run_context(self):
        """
        Law hook that provides a context manager
        in which the whole workflow is run.

        Return the hermes serve context that will
        spin up a triton and server before the
        actual condor workflow jobs are submitted
        """
        # set the triton server IP address
        # as environment variable with AFRAME prefix
        # so that condor and apptainer will tasks will
        # automatically map it into the environment
        ip = self.get_ip_address()
        os.environ["AFRAME_TRITON_IP"] = ip
        server_log = self.output_dir / "server.log"

        # TODO: figure out why serves
        # `gpus` variable does not expose
        # proper GPU ids to triton
        serve_context = serve(
            self.model_repo_dir,
            self.triton_image,
            log_file=server_log,
            wait=True,
        )

        current_gpus = os.getenv("CUDA_VISIBLE_DEVICES", "")

        # helper class to combine
        # the serve and monitor contexts
        class ServerContext:
            def __init__(self, obj):
                self.stack = ExitStack()
                self.obj = obj

            def __enter__(self):
                os.environ["CUDA_VISIBLE_DEVICES"] = self.obj.gpus
                self.stack.enter_context(serve_context)
                monitor = ServerMonitor(
                    model_name=self.obj.model_name,
                    ips="localhost",
                    filename=self.obj.output_dir
                    / f"server-stats-{self.obj.batch_size}.csv",
                    model_version=self.obj.model_version,
                    name="monitor",
                    rate=10,
                )
                time.sleep(1)
                self.stack.enter_context(monitor)

            def __exit__(self, *args):
                self.stack.close()
                os.environ["CUDA_VISIBLE_DEVICES"] = current_gpus

        return ServerContext(self)


@inherits(InferParameters)
class DeployInferRemote(InferBase):
    """
    Launch inference using remote Triton servers running as separate Condor jobs.

    This task spawns Triton inference servers as separate HTCondor jobs
    with GPU resources, then assigns inference workers to connect to those
    servers. This allows for better resource utilization and scaling.
    """

    triton_image = luigi.Parameter(
        description="Path to Triton server singularity image",
    )
    num_servers = luigi.IntParameter(
        default=1,
        description="Number of Triton servers to launch as separate Condor jobs",
    )
    gpus_per_server = luigi.IntParameter(
        default=1,
        description="Number of GPUs per Triton server",
    )
    cuda_min_memory = luigi.IntParameter(
        default=40000,
        description="Minimum CUDA memory (MB) required for server nodes",
    )
    cuda_device_name = luigi.Parameter(
        default="A100",
        description="Regex pattern to match GPU device name for servers",
    )
    server_timeout = luigi.IntParameter(
        default=600,
        description="Timeout (seconds) waiting for servers to be ready",
    )
    server_request_memory = luigi.Parameter(
        default="64 Gb",
        description="Memory to request for each Triton server job",
    )
    server_request_cpus = luigi.IntParameter(
        default=4,
        description="CPUs to request for each Triton server job",
    )

    exclude_params_branch = InferBase.exclude_params_branch | {
        "num_servers",
        "gpus_per_server",
        "cuda_min_memory",
        "cuda_device_name",
        "server_timeout",
        "server_request_memory",
        "server_request_cpus",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._server_pool = None

    @property
    def model_repo_dir(self):
        # TODO: Make this configurable via parameter
        return "/home/barmstrong/aframe_new/runs/model_repository"

    @property
    def server_output_dir(self):
        return paths().condor_dir / "triton" / "output"

    @property
    def server_pool(self):
        """Get or create the server pool for distributing workers."""
        if self._server_pool is None:
            self._server_pool = TritonServerPool(
                self.server_output_dir,
                self.num_servers,
            )
        return self._server_pool

    def workflow_requires(self):
        """Add Triton server deployment as a workflow requirement."""
        reqs = super().workflow_requires()
        reqs["triton_servers"] = TritonServerTask.req(
            self,
            num_servers=self.num_servers,
            gpus_per_server=self.gpus_per_server,
            triton_image=self.triton_image,
            model_repo_dir=self.model_repo_dir,
            cuda_min_memory=self.cuda_min_memory,
            cuda_device_name=self.cuda_device_name,
            server_timeout=self.server_timeout,
            request_memory=self.server_request_memory,
            request_cpus=self.server_request_cpus,
        )
        return reqs

    def htcondor_workflow_run_context(self):
        """
        Law hook that provides a context manager for the workflow run.

        For remote deployment, we wait for Triton servers to be ready
        and then configure workers to connect to them.
        """

        class RemoteServerContext:
            def __init__(self, obj):
                self.obj = obj

            def __enter__(self):
                # Wait for all servers to be ready
                logging.info(
                    f"Waiting for {self.obj.num_servers} Triton servers to be ready..."
                )
                self.obj.server_pool.load_servers(
                    wait_for_ready=True,
                    timeout=self.obj.server_timeout,
                )
                logging.info(
                    f"All {len(self.obj.server_pool.ready_servers)} servers ready"
                )

                # Set environment variable with comma-separated server addresses
                # Workers will use this to find available servers
                addresses = ",".join(self.obj.server_pool.grpc_addresses)
                os.environ["AFRAME_TRITON_SERVERS"] = addresses

                # For backwards compatibility, set AFRAME_TRITON_IP to first server
                if self.obj.server_pool.ready_servers:
                    first_server = self.obj.server_pool.ready_servers[0]
                    os.environ["AFRAME_TRITON_IP"] = first_server["hostname"]

            def __exit__(self, *args):
                # Servers continue running as separate Condor jobs
                # They will be cleaned up when their jobs are terminated
                pass

        return RemoteServerContext(self)

    def get_triton_address_for_branch(self) -> str:
        """
        Get the Triton server address for this branch.

        Uses deterministic assignment based on branch number for
        load balancing across servers.
        """
        server = self.server_pool.get_server_for_branch(self.branch)
        return server["grpc_address"]

    def run(self):
        """
        Run inference connecting to a remote Triton server.
        """
        from hermes.aeriel.client import InferenceClient

        from infer.data import Sequence
        from infer.main import infer
        from infer.postprocess import Postprocessor

        # Get server address for this branch
        # First try branch-based assignment, fall back to environment variable
        try:
            ip = self.get_triton_address_for_branch()
            # Extract just the hostname part for the sequence
            hostname = ip.split(":")[0]
        except RuntimeError:
            # Fall back to environment variable
            ip = os.getenv("AFRAME_TRITON_IP")
            hostname = ip

        self.tmp_dir.mkdir(exist_ok=True, parents=True)
        fname, shifts = self.branch_data
        sequence = Sequence(
            ifos=self.ifos,
            batch_size=self.batch_size,
            inference_sampling_rate=self.inference_sampling_rate,
            rate=self.rate_per_client,
            shifts=shifts,
            background_fname=fname,
            injection_set_fname=self.injection_set_fname,
            triton_address=hostname,
        )

        postprocessor = Postprocessor(
            integration_window_length=self.integration_window_length,
            inference_sampling_rate=self.inference_sampling_rate,
            cluster_window_length=self.cluster_window_length,
            psd_length=self.psd_length,
            fduration=self.fduration,
            t0=sequence.t0,
            shifts=shifts,
        )

        # Use full address with port for client
        address = ip if ":" in ip else f"{ip}:8001"
        client = InferenceClient(
            address=address,
            model_name=self.model_name,
            model_version=self.model_version,
            callback=sequence,
        )

        with client:
            outputs = infer(
                client, sequence, postprocessor, self.return_timeseries
            )

        if self.return_timeseries:
            background, foreground, background_ts, foreground_ts = outputs
            with h5py.File(self.timeseries_output, "w") as f:
                f.attrs["t0"] = postprocessor.t0
                f.attrs["shifts"] = postprocessor.shifts
                f.create_dataset("background", data=background_ts)
                if foreground_ts is None:
                    foreground_ts = np.zeros(0)
                f.create_dataset("foreground", data=foreground_ts)
        else:
            background, foreground = outputs

        background.write(self.background_output)
        foreground.write(self.foreground_output)

        # Create metadata files
        metadata = {
            "background_length": len(background),
            "foreground_length": len(foreground),
            "shifts": shifts,
            "triton_server": ip,
        }
        with open(self.metadata_output, "w") as f:
            json.dump(metadata, f)


@inherits(DeployInferLocal)
class Infer(AframeSingularityTask):
    """
    Law Task that aggregates results from
    individual condor inference jobs.

    Supports two deployment modes:
    - local: Runs Triton server on the main job node (default)
    - remote: Spawns separate Condor jobs for Triton servers with GPUs

    Use `deployment_mode = remote` to launch GPU-enabled Triton servers
    as separate HTCondor jobs for better resource utilization.
    """

    remove_tmpdir = luigi.BoolParameter(
        description="If `True`, remove directory where individual segment"
        " results are stored after aggregation. Defaults to `True`.",
        default=True,
    )
    deployment_mode = luigi.ChoiceParameter(
        choices=["local", "remote"],
        default="local",
        description="Deployment mode for Triton servers: "
        "'local' runs server on main node, "
        "'remote' spawns separate Condor GPU jobs for servers.",
    )
    # Remote deployment specific parameters
    num_servers = luigi.IntParameter(
        default=1,
        description="Number of Triton servers to launch (remote mode only)",
    )
    gpus_per_server = luigi.IntParameter(
        default=1,
        description="Number of GPUs per Triton server (remote mode only)",
    )
    cuda_min_memory = luigi.IntParameter(
        default=40000,
        description="Minimum CUDA memory (MB) for server nodes (remote mode only)",
    )
    cuda_device_name = luigi.Parameter(
        default="A100",
        description="Regex to match GPU device name for servers (remote mode only)",
    )
    server_timeout = luigi.IntParameter(
        default=600,
        description="Timeout (seconds) waiting for servers (remote mode only)",
    )
    server_request_memory = luigi.Parameter(
        default="64 Gb",
        description="Memory for each Triton server job (remote mode only)",
    )
    server_request_cpus = luigi.IntParameter(
        default=4,
        description="CPUs for each Triton server job (remote mode only)",
    )

    @property
    def default_image(self):
        return "infer.sif"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.foreground_output = self.output_dir / "foreground.hdf5"
        self.background_output = self.output_dir / "background.hdf5"
        self.zero_lag_output = self.output_dir / "0lag.hdf5"
        self.timeseries_output = self.output_dir / "timeseries.hdf5"

    def output(self):
        output = {}
        output["foreground"] = law.LocalFileTarget(self.foreground_output)
        output["background"] = law.LocalFileTarget(self.background_output)
        if self.zero_lag:
            output["zero_lag"] = law.LocalFileTarget(self.zero_lag_output)
        if self.return_timeseries:
            output["timeseries"] = law.LocalFileTarget(self.timeseries_output)
        return output

    def requires(self):
        # Choose deployment task based on mode
        if self.deployment_mode == "remote":
            return DeployInferRemote.req(
                self,
                request_memory=self.request_memory,
                request_disk=self.request_disk,
                request_cpus=self.request_cpus,
                workflow=self.workflow,
                poll_interval=0.2,
                num_servers=self.num_servers,
                gpus_per_server=self.gpus_per_server,
                cuda_min_memory=self.cuda_min_memory,
                cuda_device_name=self.cuda_device_name,
                server_timeout=self.server_timeout,
                server_request_memory=self.server_request_memory,
                server_request_cpus=self.server_request_cpus,
            )
        else:
            # Default local deployment
            return DeployInferLocal.req(
                self,
                request_memory=self.request_memory,
                request_disk=self.request_disk,
                request_cpus=self.request_cpus,
                workflow=self.workflow,
                poll_interval=0.2,
            )

    @property
    def targets(self):
        return list(self.input().collection.targets.values())

    @property
    def background_files(self):
        return np.array(
            [Path(targets["background"].path) for targets in self.targets]
        )

    @property
    def foreground_files(self):
        return np.array(
            [Path(targets["foreground"].path) for targets in self.targets]
        )

    @property
    def metadata_files(self):
        return np.array(
            [Path(targets["metadata"].path) for targets in self.targets]
        )

    @property
    def timeseries_files(self):
        if self.return_timeseries:
            return np.array(
                [Path(targets["timeseries"].path) for targets in self.targets]
            )
        return None

    def get_metadata(self):
        """
        Read in shift and length metadata from the metadata
        files created by each `DeployInferLocal` condor job.
        This data is read from the metadata files rather than
        the hdf5 files because the read operation is O(1000)
        times faster this way
        """
        files = self.metadata_files
        num_files = len(files)
        background_lengths = np.zeros(num_files)
        foreground_lengths = np.zeros(num_files)
        shifts = np.zeros((num_files, len(self.shifts)))
        for i, f in enumerate(files):
            with open(f, "r") as f:
                data = json.load(f)
            background_lengths[i] = data["background_length"]
            foreground_lengths[i] = data["foreground_length"]
            shifts[i] = data["shifts"]

        return background_lengths, foreground_lengths, shifts

    def aggregate_timeseries(self):
        index = []
        with h5py.File(self.timeseries_output, "w") as f:
            ts_group = f.create_group("timeseries")
            for ts in self.timeseries_files:
                with h5py.File(ts, "r") as g:
                    t0 = g.attrs["t0"]
                    shifts = g.attrs["shifts"]
                    background = g["background"][:]
                    foreground = g["foreground"][:]
                    ts_id = f"{t0}_{shifts}"
                    subgroup = ts_group.create_group(ts_id)
                    subgroup.create_dataset("background", data=background)
                    subgroup.create_dataset("foreground", data=foreground)
                    index.append((t0, shifts, f"/timeseries/{ts_id}"))

            # Create an index to make it easier to look up
            # specific segments and shifts
            dtype = np.dtype(
                [
                    ("t0", np.float64),
                    ("shifts", np.int32, (len(shifts),)),
                    ("path", h5py.string_dtype(encoding="utf-8")),
                ]
            )
            index_array = np.array(index, dtype=dtype)
            f.create_dataset("index", data=index_array)

    def run(self):
        import shutil

        from ledger.events import EventSet, RecoveredInjectionSet

        # separate 0lag and background events into different files
        background_lengths, foreground_lengths, shifts = self.get_metadata()
        zero_lag = np.array(
            [all(shift == [0] * len(self.ifos)) for shift in shifts]
        )

        zero_lag_files = self.background_files[zero_lag]
        back_files = self.background_files[~zero_lag]
        zero_lag_length = sum(background_lengths[zero_lag])
        background_length = sum(background_lengths[~zero_lag])
        foreground_length = sum(foreground_lengths)
        foreground_mask = foreground_lengths > 0

        logging.info("Aggregating background files")
        EventSet.aggregate(
            back_files,
            self.background_output,
            clean=False,
            length=background_length,
        )
        logging.info("Aggregating foreground files")
        RecoveredInjectionSet.aggregate(
            self.foreground_files[foreground_mask],
            self.foreground_output,
            clean=False,
            length=foreground_length,
        )
        if len(zero_lag_files) > 0:
            logging.info("Aggregating zero lag files")
            EventSet.aggregate(
                zero_lag_files,
                self.zero_lag_output,
                clean=False,
                length=zero_lag_length,
            )
        if self.return_timeseries:
            logging.info("Aggregating timeseries files")
            self.aggregate_timeseries()

        # Sort background events for later use.
        # TODO: any benefit to sorting foreground for SV calculation?
        if len(back_files) > 0:
            background = EventSet.read(self.background_output)
            background = background.sort_by("detection_statistic")
            background.write(self.background_output)

        if self.remove_tmpdir:
            shutil.rmtree(self.output_dir / "tmp")
