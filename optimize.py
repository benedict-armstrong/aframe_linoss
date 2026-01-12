#!/usr/bin/env python3
"""Optimize JAX model using Model Navigator."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxlib import xla_extension
import tensorflow  # Required for JAX optimization pipeline

import model_navigator as nav

# Set TensorFlow GPU memory growth to avoid conflicts
gpus = tensorflow.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)


def load_model():
    """Load your JAX model from the exported file."""
    with open(
        "/home/barmstrong/ssm_bench/exported_model.jax",
        "rb",
    ) as f:
        model_bytes = f.read()

    model = jax.export.deserialize(model_bytes)
    return model


def get_model_and_params():
    """
    Create a forward function and parameters for Model Navigator.

    Model Navigator expects:
    - model: A callable JAX forward function
    - model_params: Model parameters (weights)
    """
    exported_model = load_model()

    # Wrap the exported model's call method as the forward function
    # Model Navigator expects a function signature like:  predict(inputs, params)
    def predict(inputs, params):
        # Your model may not need params if already serialized
        # Transpose input to match your model's expected format
        inputs = inputs # (B, 8192, 2)
        result = exported_model.call(inputs)
        return result[..., 1:]  # Return only the discriminator output

    # If your model is already serialized with weights, params can be empty
    params = {}

    return predict, params


def get_dataloader():
    """
    Create a dataloader with representative samples.

    Model Navigator uses these samples for:
    - Model export/conversion
    - Correctness testing
    - Profiling
    """
    # Create samples matching your input shape:  (batch_size, num_features, sequence_length)
    # Based on your code:  input shape is (128, 8192, 2) after transpose becomes (128, 2, 8192)
    batch_sizes = [128]  # Test various batch sizes
    dataloader = []

    for batch_size in batch_sizes:
        # Create random input data matching your model's expected input
        sample = np.random.randn(batch_size, 8192, 2).astype(np.float32)
        dataloader.append(sample)

    return dataloader


def get_verify_function():
    """Define verification function to compare model outputs."""

    def verify_func(ys_runner, ys_expected):
        for y_runner, y_expected in zip(ys_runner, ys_expected):
            if not all(
                np.allclose(a, b, rtol=1.0e-3, atol=1.0e-3)
                for a, b in zip(y_runner.values(), y_expected.values())
            ):
                return False
        return True

    return verify_func


def main():
    """Run optimization and save the package."""

    # Load model and prepare dataloader
    model, params = get_model_and_params()
    dataloader = get_dataloader()
    verify_func = get_verify_function()

    # Run optimization
    # This will export, convert, test correctness, and profile the model
    package = nav.jax.optimize(
        model=model,
        model_params=params,
        dataloader=dataloader,
        verify_func=verify_func,
        batching=False,  # Enable batching on first dimension
        verbose=True,  # Enable verbose logging
        # Optional: specify target formats explicitly
        # target_formats=(nav.Format.TF_SAVEDMODEL, nav.Format. ONNX, nav.Format. TENSORRT),
        # Optional: customize optimization profile
        # optimization_profile=nav.OptimizationProfile(max_batch_size=128),
    )

    # Save the optimized package
    nav.package.save(package, "aframe_optimized.nav", override=True)
    print("Package saved to aframe_optimized.nav")

    # Optional: Create Triton model repository directly
    import pathlib

    try:
        nav.triton.model_repository.add_model_from_package(
            model_repository_path=pathlib.Path("model_repository"),
            model_name="aframe",
            package=package,
        )
        print("Model repository created at ./model_repository")
    except Exception as e:
        print(f"Could not create model repository: {e}")


if __name__ == "__main__":
    main()
