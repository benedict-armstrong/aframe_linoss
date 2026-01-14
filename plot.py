import requests
import time
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

# get node from command line argument
parser = argparse.ArgumentParser(description="Triton Metrics Plotter")
parser.add_argument(
    "--node",
    type=str,
    required=True,
    help="The node address to fetch metrics from (e.g., g145)",
)
args = parser.parse_args()


# --- Configuration ---
METRICS_URL = f"http://{args.node}.internal.cluster.is.localnet:8002/metrics"
OUTPUT_DIR = "triton_plots"
POLL_INTERVAL = 1.0  # seconds

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Store history for plotting
history = defaultdict(list)
timestamps = []


def parse_metrics(text):
    """Very simple parser for the Triton metrics format."""
    results = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue

        # Split into key and value
        try:
            parts = line.rsplit(" ", 1)
            metric_full = parts[0]
            value = float(parts[1])
            results[metric_full] = value
        except (IndexError, ValueError):
            continue
    return results


def update_plots():
    plt.close("all")  # Clear memory

    # We will group plots by metric type (ignoring labels for the filename)
    # Example: nv_inference_pending_request_count
    grouped_metrics = defaultdict(list)
    for key in history.keys():
        base_name = key.split("{")[0]
        grouped_metrics[base_name].append(key)

    for base_name, full_keys in grouped_metrics.items():
        plt.figure(figsize=(10, 5))
        for k in full_keys:
            # Shorten label for the legend
            label = k.replace(base_name, "").strip("{}")
            plt.plot(
                timestamps, history[k], label=label if label else base_name
            )

        plt.title(f"Metric: {base_name}")
        plt.xlabel("Seconds Elapsed")
        plt.ylabel("Value")
        if len(full_keys) > 1:
            plt.legend(loc="upper left", fontsize="small")

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}.png"))


print(
    f"Polling {METRICS_URL} every {POLL_INTERVAL}s. Plots saving to '{OUTPUT_DIR}'..."
)

start_time = time.time()
try:
    while True:
        try:
            response = requests.get(METRICS_URL, timeout=2)
            metrics = parse_metrics(response.text)

            elapsed = round(time.time() - start_time, 1)
            timestamps.append(elapsed)

            for k, v in metrics.items():
                history[k].append(v)

            update_plots()
            print(f"[{elapsed}s] Metrics updated and plots saved.")

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(POLL_INTERVAL)
except KeyboardInterrupt:
    print("\nStopped by user.")
