"""
Triton Inference Server Condor task for running GPU-accelerated inference servers.

This module provides a Luigi/law task that deploys Triton Inference Servers
as HTCondor jobs, enabling distributed GPU inference for aframe pipelines.
"""

import json
import os
import socket
import time
from pathlib import Path

import law
import luigi

from aframe.config import paths
from aframe.parameters import PathParameter
from aframe.tasks.data.condor.gpu import StaticMemoryGPUWorkflow


class TritonServerParameters(law.Task):
    """
    Parameters for configuring Triton Inference Server deployment.
    """

    triton_image = luigi.Parameter(
        description="Path to Triton server singularity image",
    )
    model_repo_dir = PathParameter(
        description="Path to the model repository directory",
    )
    num_servers = luigi.IntParameter(
        default=1,
        description="Number of Triton servers to launch",
    )
    gpus_per_server = luigi.IntParameter(
        default=1,
        description="Number of GPUs to allocate per server",
    )
    server_timeout = luigi.IntParameter(
        default=3600,
        description="Timeout in seconds for server to be ready",
    )
    grpc_port = luigi.IntParameter(
        default=8001,
        description="Base gRPC port for Triton server",
    )
    http_port = luigi.IntParameter(
        default=8000,
        description="Base HTTP port for Triton server",
    )
    metrics_port = luigi.IntParameter(
        default=8002,
        description="Base metrics port for Triton server",
    )


class TritonServerTask(
    StaticMemoryGPUWorkflow,
    law.LocalWorkflow,
    TritonServerParameters,
):
    """
    Law task to deploy Triton Inference Servers via HTCondor.

    This task launches one or more Triton servers on GPU nodes,
    each running as a separate Condor job. The servers are
    long-running background processes that worker tasks can
    connect to for inference.
    """

    condor_directory = PathParameter(default=paths().condor_dir / "triton")
    request_memory = luigi.Parameter(default="64 Gb")
    request_cpus = luigi.IntParameter(default=4)
    cuda_min_memory = luigi.IntParameter(default=40000)
    cuda_device_name = luigi.Parameter(default="A100")

    # File to store server info for workers
    server_info_file = PathParameter(default=None)

    exclude_params_req = StaticMemoryGPUWorkflow.exclude_params_req | {
        "server_info_file",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.server_info_file is None:
            self.server_info_file = self.condor_directory / "server_info.json"

        # Set GPUs per server
        self.request_gpus = self.gpus_per_server

        # Ensure output directories exist
        self.htcondor_log_dir.touch()
        self.htcondor_output_directory().touch()

    @property
    def name(self):
        return "tritonserver"

    @property
    def output_dir(self):
        return self.condor_directory / "output"

    def create_branch_map(self):
        """Create branch map with one branch per server."""
        return {i: i for i in range(self.num_servers)}

    def output(self):
        """Output target indicating server is running."""
        return {
            "ready": law.LocalFileTarget(
                self.output_dir / f"server-{self.branch}.ready"
            ),
            "info": law.LocalFileTarget(
                self.output_dir / f"server-{self.branch}.json"
            ),
        }

    def get_server_hostname(self) -> str:
        """Get the cluster-internal hostname for this node."""
        hostname = socket.gethostname()
        # Add cluster-internal domain if needed
        if not hostname.endswith(".internal.cluster.is.localnet"):
            hostname = hostname + ".internal.cluster.is.localnet"
        return hostname

    def run(self):
        """
        Run the Triton Inference Server.

        This starts a Triton server process and waits for it to become ready,
        then writes server info and keeps running until terminated.
        """
        import subprocess
        import signal
        import sys

        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True, parents=True)

        hostname = self.get_server_hostname()
        server_id = self.branch

        # Calculate ports for this server instance
        grpc_port = self.grpc_port + server_id
        http_port = self.http_port + server_id
        metrics_port = self.metrics_port + server_id

        # Write server info
        server_info = {
            "hostname": hostname,
            "server_id": server_id,
            "grpc_port": grpc_port,
            "http_port": http_port,
            "metrics_port": metrics_port,
            "grpc_address": f"{hostname}:{grpc_port}",
            "http_address": f"{hostname}:{http_port}",
            "status": "starting",
        }

        info_file = self.output()["info"].path
        with open(info_file, "w") as f:
            json.dump(server_info, f)

        # Build Triton server command
        triton_cmd = [
            "singularity",
            "run",
            "--nv",
            self.triton_image,
            "tritonserver",
            f"--model-repository={self.model_repo_dir}",
            f"--grpc-port={grpc_port}",
            f"--http-port={http_port}",
            f"--metrics-port={metrics_port}",
            "--log-verbose=1",
        ]

        # Start the server process
        log_file = self.output_dir / f"server-{server_id}.log"
        with open(log_file, "w") as log:
            process = subprocess.Popen(
                triton_cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        # Wait for server to be ready
        ready = self._wait_for_server(hostname, grpc_port)

        if ready:
            # Update server info to indicate ready status
            server_info["status"] = "ready"
            server_info["pid"] = process.pid
            with open(info_file, "w") as f:
                json.dump(server_info, f)

            # Create ready marker file
            ready_file = self.output()["ready"].path
            Path(ready_file).touch()

            # Set up signal handlers for clean shutdown
            def signal_handler(sig, frame):
                process.terminate()
                process.wait()
                sys.exit(0)

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            # Keep running until terminated
            process.wait()
        else:
            process.terminate()
            raise RuntimeError(
                f"Triton server {server_id} failed to start within timeout"
            )

    def _wait_for_server(self, hostname: str, port: int) -> bool:
        """
        Wait for the Triton server to become ready.

        Args:
            hostname: Server hostname
            port: gRPC port number

        Returns:
            True if server is ready, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < self.server_timeout:
            try:
                # Try to connect to the gRPC port
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((hostname, port))
                sock.close()
                if result == 0:
                    # Additional check: wait a bit for full initialization
                    time.sleep(5)
                    return True
            except (socket.error, socket.timeout):
                pass
            time.sleep(10)
        return False


class TritonServerPool:
    """
    Manages a pool of Triton servers and distributes workers across them.

    This class provides round-robin assignment of workers to servers
    and tracks server availability.
    """

    def __init__(self, server_info_dir: Path, num_servers: int):
        """
        Initialize the server pool.

        Args:
            server_info_dir: Directory containing server info JSON files
            num_servers: Expected number of servers
        """
        self.server_info_dir = Path(server_info_dir)
        self.num_servers = num_servers
        self._servers = []
        self._assignment_counter = 0

    def load_servers(self, wait_for_ready: bool = True, timeout: int = 600):
        """
        Load server information from info files.

        Args:
            wait_for_ready: If True, wait for all servers to be ready
            timeout: Timeout in seconds to wait for servers

        Raises:
            RuntimeError: If servers are not ready within timeout
        """
        start_time = time.time()

        while True:
            self._servers = []
            all_ready = True

            for i in range(self.num_servers):
                info_file = self.server_info_dir / f"server-{i}.json"
                ready_file = self.server_info_dir / f"server-{i}.ready"

                if info_file.exists():
                    with open(info_file, "r") as f:
                        info = json.load(f)

                    if ready_file.exists():
                        info["ready"] = True
                    else:
                        info["ready"] = False
                        all_ready = False

                    self._servers.append(info)
                else:
                    all_ready = False

            if all_ready and len(self._servers) == self.num_servers:
                return

            if not wait_for_ready:
                return

            if time.time() - start_time > timeout:
                raise RuntimeError(
                    f"Timeout waiting for {self.num_servers} servers. "
                    f"Only {len([s for s in self._servers if s.get('ready')])} ready."
                )

            time.sleep(10)

    def get_next_server(self) -> dict:
        """
        Get the next server in round-robin order.

        Returns:
            Server info dictionary
        """
        if not self._servers:
            raise RuntimeError(
                "No servers available. Call load_servers() first."
            )

        ready_servers = [s for s in self._servers if s.get("ready", False)]
        if not ready_servers:
            raise RuntimeError("No ready servers available.")

        server = ready_servers[self._assignment_counter % len(ready_servers)]
        self._assignment_counter += 1
        return server

    def get_server_for_branch(self, branch: int) -> dict:
        """
        Get the server assigned to a specific branch number.

        This provides deterministic assignment based on branch number.

        Args:
            branch: Branch number

        Returns:
            Server info dictionary
        """
        if not self._servers:
            raise RuntimeError(
                "No servers available. Call load_servers() first."
            )

        ready_servers = [s for s in self._servers if s.get("ready", False)]
        if not ready_servers:
            raise RuntimeError("No ready servers available.")

        return ready_servers[branch % len(ready_servers)]

    @property
    def servers(self) -> list:
        """Return list of all servers."""
        return self._servers

    @property
    def ready_servers(self) -> list:
        """Return list of ready servers."""
        return [s for s in self._servers if s.get("ready", False)]

    @property
    def grpc_addresses(self) -> list:
        """Return list of gRPC addresses for ready servers."""
        return [s["grpc_address"] for s in self.ready_servers]
