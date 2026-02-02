import json
import sys

# sys.path.append(
#     "/lustre/home/barmstrong/test_install/.venv/lib64/python3.12/site-packages"
# )

import torch
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from pathlib import Path

from .ssm_bench import InferenceModel

import triton_python_backend_utils as pb_utils

BATCH_SIZE = 512


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Absolute model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # print(help("modules"))

        try:
            print(
                f"Starting model on {args['model_instance_device_id']} of kind {args['model_instance_kind']}",
                flush=True,
            )
        except Exception:
            pass

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get the GPU device ID assigned by Triton for this instance
        self.device_id = int(args.get("model_instance_device_id", "0"))
        self.device = jax.devices("gpu")[self.device_id]
        print(f"Using JAX device: {self.device}", flush=True)

        config_file = Path(
            "/lustre/home/barmstrong/aframe_new/runs/model_repository/aframe/1/config.yaml"
        )
        weights_file = Path(
            "/lustre/home/barmstrong/aframe_new/runs/model_repository/aframe/1/weights.eqx"
        )

        cfg = OmegaConf.load(config_file)
        model_cfg = OmegaConf.to_container(cfg["model"], resolve=True)
        in_shape = tuple([2048 * 8, 2])
        out_shape = {
            "label": (2,),
            "snr": (1,),
            "mass_params": (4,),
        }
        input_dtype = jax.numpy.float32

        print("Loading model...", flush=True)

        self.model = InferenceModel(
            model_cfg=model_cfg,
            in_shape=in_shape,
            out_shape=out_shape,
            input_dtype=input_dtype,
            checkpoint_path=weights_file,
            seed=0,
        )

        # run a dummy inference to make sure model is jit compiled
        # Place tensors on the correct GPU device
        input = jax.device_put(jnp.ones((BATCH_SIZE, 8192, 2)), self.device)
        result = self.model(input)
        print(
            "Inizialized model result: ",
            result,
            flush=True,
        )

        # Get OUTPUT1 configuration
        output_config = pb_utils.get_output_config_by_name(
            model_config, "discriminator"
        )

        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config["data_type"]
        )

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        outputs = []

        # Every Python backend must iterate over every one of the requests and
        # create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            input_tensor = pb_utils.get_input_tensor_by_name(request, "X")

            # Place input on the correct GPU device assigned to this instance
            input = jax.device_put(
                jax.dlpack.from_dlpack(
                    torch.from_dlpack(input_tensor.to_dlpack())
                ).transpose(0, 2, 1),
                self.device,
            )

            output = self.model(input)
            result = jnp.array(output).reshape(-1, 1)
            outputs.append(result)

        responses = []
        for output in outputs:
            # shape (B, 1)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor = pb_utils.Tensor.from_dlpack(
                "discriminator", output.astype(self.output_dtype)
            )

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor],
            )
            responses.append(inference_response)

            # You should return a list of pb_utils.InferenceResponse. Length
            # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
