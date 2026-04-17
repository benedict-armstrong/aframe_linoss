"""CLI entry point for regression training.

Run:
    uv run python -m train.regression_cli fit --config regression.yaml

The YAML must wire up a LightningDataModule subclass (e.g.
RegressionTimeDomainDataset) and a LightningModule subclass (e.g.
LitLinOSSGaussianNLL or LitS4DGaussianNLL).
"""

import torch
from lightning.pytorch.cli import LightningCLI


def main():
    torch.set_float32_matmul_precision("high")
    LightningCLI(save_config_callback=None)


if __name__ == "__main__":
    main()
