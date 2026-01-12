# Aframe
Detecting compact binary mergers from gravitational wave strain data using neural networks. 
See our [documentation](https://ml4gw-aframe.readthedocs.io/en/latest/index.html) for information on how to get started.

Please cite ["A machine-learning pipeline for real-time detection of gravitational waves from compact binary coalescences"](https://arxiv.org/abs/2403.18661) 
if you use `Aframe` software in your work.

# Notes

Start triton server manually by

```bash
a100 # request a100 node
singularity shell  --nv /fast/barmstrong/container/aframe/tritonserver_25.06-py3_jax.sif
tritonserver --model-repository=runs/model_repository --log-verbose=1
```