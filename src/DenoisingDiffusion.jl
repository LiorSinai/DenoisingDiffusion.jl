module DenoisingDiffusion

import Base.show, Base.eltype

using Flux
import Flux._big_show, Flux._show_children
using ProgressMeter
using Printf
using BSON
using Random
import NNlib: batched_mul

include("GaussianDiffusion.jl")
include("guidance.jl")
include("ddim.jl")
include("ddim_guidance.jl")
include("train.jl")

export GaussianDiffusion
export linear_beta_schedule, cosine_beta_schedule
export q_sample, q_posterior_mean_variance
export p_sample, p_sample_loop, p_sample_loop_all, p_losses, predict_start_from_noise
export ddim_sample, ddim_sample_loop

include("models/embed.jl")
export SinusoidalPositionEmbedding
include("models/ConditionalChain.jl")
export AbstractParallel, ConditionalChain, ConditionalSkipConnection
include("models/blocks.jl")
export ResBlock, ConvEmbed
include("models/attention.jl")
include("models/batched_mul_4d.jl")
export MultiheadAttention, scaled_dot_attention, batched_mul
include("models/UNetFixed.jl")
export UNetFixed
include("models/UNet.jl")
include("models/UNetConditioned.jl")
export UNet, UNetConditioned

end # module
