# DenoisingDiffusion.jl

A pure Julia implementation of denoising diffusion probabilistic models as popularised in [Denoising Diffusion Probabilistic Models by Jonathan Ho, Ajay Jain and Pieter Abbeel (2020)](https://arxiv.org/abs/2006.11239)

For detailed examples please the notebooks at the corresponding project at [github.com/LiorSinai/DenoisingDiffusion-examples](https://github.com/LiorSinai/DenoisingDiffusion-examples). The notebooks were originally part of this repository but were removed using [git-filter-repo](https://github.com/newren/git-filter-repo) to make this repository more lightweight.

For an explanation of the diffusion process and the code please see my blog posts at [liorsinai.github.io](https://liorsinai.github.io/coding/2022/12/03/denoising-diffusion-1-spiral.html).

## Overview 
### Unconditioned sampling

<p align="center">
  <img src="images/ddpm.png"/>
</p>

Denoising diffusion starts from an image of pure noise and gradually removes this noise across many time steps, resulting in an image representative of the training data.
At each time step a model predicts the noise to be removed in order to reach the final image `x_0` on the final time step `t=0` from the current time step `t`.
This is then used to denoise the image by one time step and a better estimate of `x_0` is made on the next time step.

<p align="center">
  <img src="images/numbers_reverse.gif" width="45%" style="padding:5px"/>
  <img src="images/numbers_estimate.gif" width="45%"  style="padding:5px"/> 
  <p style="text-align:center">Reverse process (left) and final image estimate (right). <code>i=T-t</code></p>
</p>

The above image shows this process with a trained model for number generation.
The final image estimates gradually improve throughout the process and coincides with the actual image on the final time step.

### Conditioned sampling with classifier free guidance
<p align="center">
  <img src="images/2d_reverse_guided.gif" width="400" height="400"/>
  <p style="text-align:center">Classifier free guidance</p>
</p>

It is possible to direct the outcome using classifier free guidance as introduced in 
[Classifier-Free Diffusion Guidance by Jonathan Ho and Tim Salimans (2022)](https://arxiv.org/abs/2207.12598).
In this mode a label as well as the timestep is passed to the model. 
Two candidates of the noise to be removed are generated at each timestep: unconditioned noise made using a generic label (label=1) and conditioned noise made using the target label.
The noise that is removed is then given by a weighted combination of the two:
```
noise = ϵ_uncond + guidance_scale * (ϵ_cond - ϵ_uncond)
```
Where `guidance_scale >= 1`. The difference `(ϵ_cond - ϵ_uncond)` represents a rough gradient.

The original paper uses `ϵ_cond + guidance_scale * (ϵ_cond - ϵ_uncond)` but using the baseline as `ϵ_uncond` instead allows it to be cancelled and skipped for the special case of `guidance_scale = 1`.

## Module 

The main export is the `GaussianDiffusion` struct and associated functions.
Various models and layers are included. 

### Layers

Skip connection:
- `ConditionalSkipConnection`: a skip connection which can pass multiple arguments to its layers.

Embedding:
- `SinusoidalPositionEmbedding`: a non-trainable matrix of position embeddings based on sine and cosine waves.

Convolution:
- `ConvEmbed`: calculate a convolution on the first input and add embeddings from the second input.
- `ResBlock`: A `ConvEmbed` block followed by convolution with a skip connection past both.
- `MultiheadAttention`: multiheaded attention with a convolution layer which calculates the key, query and value.

### Models

- `ConditionalChain`: based on `Flux.Chain`. It can handle multiple inputs where the first input is given priority. Uses
- UNet: Two versions of UNets (convolutional autoencoder) are available, `UNet` and `UNetFixed`.
  - `UNet` is flexible and can have an arbitrary number of downsample/upsample pairs (more than five is not advisable). It is based on nested skip connections.
  - `UNetFixed` has a linear implementation. 
It has three downsample/upsample pairs and three middle layers with a total of 16 layers. The default configuration `UNetFixed(1, 8, 100)` will have approximately 150,000 parameters. 
  - A `UNet` model made to the same specifications as `UNetFixed` is 100% equivalent. 
  - About 50% of these parameters are in the middle layer - 24% in the attention layer alone.
  - For both models, every doubling of the `model_channels` will approximately quadruple the number of parameters because the convolution layer size is proportional to the square of the dimension.

### GPU compatibility

This repository is fully compatible with GPU training and inference using Flux. 

An important caveat is that all generated matrices needs to be on the same device as the model.
For example, in each step of the reverse process (denoising) noise is generated and added to provide diversity in the results.
In the forward process (diffusion) noise is generated to make the sample more noisy.
This noise is generated on the CPU but if the model is on the GPU it needs to be transferred there.

The same is true of functions that automatically generate the `timesteps` vector.

This repository requires the user to manually specify the device with the `to_device` key word argument. It can either be `Flux.cpu` or `Flux.gpu`. The default is `Flux.cpu`.

Future work: automatically determine if the model is on the GPU.

## Examples

To run the examples: 
```
julia  --threads auto examples\\train_images.jl
```

Or start the Julia REPL and run it interactively.

There are three use cases:
- Spiral (2 values per data point).
- Numbers (28&times;28=784 values per data point.)
- Pokemon (48&times;48&times;3=6912 values per data point.)

The spiral use case requires approximately 1,000 parameters. The number generation requires at least 100 times this, and the Pokemon possibly more. So far, satisfying results for the Pokemon have not been achieved.
See however [This Pokémon Does Not Exist](https://huggingface.co/spaces/ronvolutional/ai-pokemon-card)
for an example trained on 1.3 billion parameter model.

## Fréchet LeNet Distances (FLD)

For number generation the [Frechet Inception Distance (FID)](https://arxiv.org/abs/1706.08500) is cumbersome. 
The [Inception V3](https://pytorch.org/hub/pytorch_vision_inception_v3/) model has 27.1 million parameters
which is overkill for number generation. Instead the simpler Fréchet LeNet Distance is proposed.
This uses the same calculation except with a smaller [LeNet model](https://github.com/FluxML/model-zoo/blob/master/vision/conv_mnist/conv_mnist.jl) with approximately 44,000 parameters.
The output layer has 84 values as opposed to Inception V3's 2048.

No pretrained weights are necessary because the LeNet model can be very easily trained on a CPU.
However results will not be standardised.

Example values are:

| Model           | Parameters | FLD   | Notes |
| ---             | ---        | ---   | ---   |
| training data   | 0          | 0.5   |       |
| UNetConditioned | 622, 865   | 7.0   | Guidance with $\gamma=2$ and 1000 samples per label  | 
| UNet            | 376,913    | 18.3  | No attention layer | 
| UNet            | 602,705    | 23.9  |       |
| UNet            | 602,705    | 26.3  | DDIM $\tau_n = 20; \eta=1$ |
| Random          | 0          | >337  |        |

The loss is Mean Squared Error. All models were trained for 15 epochs.

## Installation

Download the GitHub repository (it is not registered). Then in the Julia REPL:
```
julia> ] #enter package mode
(@v1.x) pkg> dev path\\to\\DenoisingDiffusion
julia> using Revise # allows dynamic edits to code
julia> using DenoisingDiffusion
```

Optionally, tests can be run with:
```
(@v1.x) pkg> test DenoisingDiffusion
```

This repository uses FastAi's [nbdev](https://nbdev.fast.ai/tutorials/git_friendly_jupyter.html) to manage the Jupyter Notebooks for Git. This requires a Python installation of nbdev. To avoid using it, follow the steps in .gitconfig.

## Task list

- [x] Self-attention blocks.
- [x] DDIM for more efficient and faster image generation.
- [x] Guided diffusion.
