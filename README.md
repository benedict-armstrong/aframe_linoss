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
tritonserver --model-repository=runs/model_repository
```

Build triton server container by first getting a suitable node (to avoid proc ... error) and then running

```bash
condor_submit_bid 25 -i -append 'request_memory=81920' -append 'request_cpus=10' -append 'request_disk=100G' -append '+BypassLXCfs="true"'
singularity build --nv /fast/barmstrong/container/aframe/tritonserver_25.06-py3_jax_2.sif runs/jax_triton_inference_server.def
```