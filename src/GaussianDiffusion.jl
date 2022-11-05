using Flux.CUDA

"""
    GaussianDiffusion(V::DataType, βs, data_shape, denoise_fn)

A Gaussian Diffusion Probalistic Model (DDPM) as introduced in "Denoising Diffusion Probabilistic Models" by Ho et. al (https://arxiv.org/abs/2006.11239).
"""
struct GaussianDiffusion{V<:AbstractVector}
    num_timesteps::Int
    data_shape::NTuple
    denoise_fn

    βs::V
    αs::V
    α_cumprods::V
    α_cumprod_prevs::V

    sqrt_α_cumprods::V
    sqrt_one_minus_α_cumprods::V
    sqrt_recip_α_cumprods::V
    sqrt_recip_α_cumprods_minus_one::V
    posterior_variance::V
    posterior_log_variance_clipped::V
    posterior_mean_coef1::V
    posterior_mean_coef2::V
end

eltype(::Type{<:GaussianDiffusion{V}}) where {V} = V

Flux.@functor GaussianDiffusion
Flux.trainable(g::GaussianDiffusion) = (g.denoise_fn,)

function Base.show(io::IO, diffusion::GaussianDiffusion)
    V = typeof(diffusion).parameters[1]
    print(io, "GaussianDiffusion{$V}(")
    print(io, "num_timesteps=$(diffusion.num_timesteps)")
    print(io, ", data_shape=$(diffusion.data_shape)")
    print(io, ", denoise_fn=$(diffusion.denoise_fn)")
    num_buffers = 12
    buffers_size = Base.format_bytes(Base.summarysize(diffusion.βs) * num_buffers)
    print(io, ", buffers_size=$buffers_size")
    print(io, ")")
end

function GaussianDiffusion(V::DataType, βs::AbstractVector, data_shape::NTuple, denoise_fn)
    αs = 1 .- βs
    α_cumprods = cumprod(αs)
    α_cumprod_prevs = [1, (α_cumprods[1:end-1])...]

    sqrt_α_cumprods = sqrt.(α_cumprods)
    sqrt_one_minus_α_cumprods = sqrt.(1 .- α_cumprods)
    sqrt_recip_α_cumprods = 1 ./ sqrt.(α_cumprods)
    sqrt_recip_α_cumprods_minus_one = sqrt.(1 ./ α_cumprods .- 1)

    posterior_variance = βs .* (1 .- α_cumprod_prevs) ./ (1 .- α_cumprods)
    posterior_log_variance_clipped = log.(max.(posterior_variance, 1e-20))

    posterior_mean_coef1 = βs .* sqrt.(α_cumprod_prevs) ./ (1 .- α_cumprods)
    posterior_mean_coef2 = (1 .- α_cumprod_prevs) .* sqrt.(αs) ./ (1 .- α_cumprods)

    GaussianDiffusion{V}(
        length(βs),
        data_shape,
        denoise_fn,
        βs,
        αs,
        α_cumprods,
        α_cumprod_prevs,
        sqrt_α_cumprods,
        sqrt_one_minus_α_cumprods,
        sqrt_recip_α_cumprods,
        sqrt_recip_α_cumprods_minus_one,
        posterior_variance,
        posterior_log_variance_clipped,
        posterior_mean_coef1,
        posterior_mean_coef2
    )
end

"""
    linear_beta_schedule(num_timesteps, β_start=0.0001f0, β_end=0.02f0)
"""
function linear_beta_schedule(num_timesteps::Int, β_start=0.0001f0, β_end=0.02f0)
    scale = convert(typeof(β_start), 1000 / num_timesteps)
    β_start *= scale
    β_end *= scale
    collect(range(β_start, β_end; length=num_timesteps))
end

"""
    cosine_beta_schedule(num_timesteps, s=0.008)

Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models" by Nichol, Dhariwal (https://arxiv.org/abs/2102.09672)
"""
function cosine_beta_schedule(num_timesteps::Int, s=0.008)
    t = range(0, num_timesteps; length=num_timesteps + 1)
    α_cumprods = (cos.((t / num_timesteps .+ s) / (1 + s) * π / 2)) .^ 2
    α_cumprods = α_cumprods / α_cumprods[1]
    βs = 1 .- α_cumprods[2:end] ./ α_cumprods[1:(end-1)]
    clamp!(βs, 0, 0.999)
    βs
end

## extract input[idxs] and reshape for broadcasting across a batch.
function _extract(input, idxs::AbstractVector{Int}, shape::NTuple)
    reshape(input[idxs], (repeat([1], length(shape) - 1)..., :))
end


"""
    q_sample(diffusion, x_start, timesteps, noise)
    q_sample(diffusion, x_start, timesteps; to_device=cpu)

The forward process ``q(x_t | x_0)``. Diffuse the data for a given number of diffusion steps.
"""
function q_sample(diffusion::GaussianDiffusion, x_start::AbstractArray, timesteps::AbstractVector{Int}, noise::AbstractArray)
    coeff1 = _extract(diffusion.sqrt_α_cumprods, timesteps, size(x_start))
    coeff2 = _extract(diffusion.sqrt_one_minus_α_cumprods, timesteps, size(x_start))
    coeff1 .* x_start + coeff2 .* noise
end

function q_sample(diffusion::GaussianDiffusion, x_start::AbstractArray, timesteps::AbstractVector{Int}; to_device=cpu)
    T = eltype(eltype(diffusion))
    noise =  randn(T, size(x_start)) |> to_device
    timesteps = timesteps |> to_device
    q_sample(diffusion, x_start, timesteps, noise)
end

function q_sample(diffusion::GaussianDiffusion, x_start::AbstractArray{T, N}, timestep::Int; to_device=cpu) where {T, N}
    timesteps = fill(timestep, size(x_start, N)) |> to_device
    q_sample(diffusion, x_start, timesteps; to_device=to_device)
end

"""
    q_posterior_mean_variance(diffusion, x_start, x_t, timesteps)

Compute the mean and variance for the ``q_{posterior}(x_{t-1} | x_t, x_0) = q(x_t | x_{t-1}, x_0) q(x_{t-1} | x_0) / q(x_t | x_0)``
where `x_0 = x_start`. 
The ``q_{posterior}`` is a Bayesian estimate of the reverse process ``p(x_{t-1} | x_{t})`` where ``x_0`` is known.
"""
function q_posterior_mean_variance(diffusion::GaussianDiffusion, x_start::AbstractArray, x_t::AbstractArray, timesteps::AbstractVector{Int})
    coeff1 = _extract(diffusion.posterior_mean_coef1, timesteps, size(x_t))
    coeff2 = _extract(diffusion.posterior_mean_coef2, timesteps, size(x_t))
    posterior_mean = coeff1 .* x_start + coeff2 .* x_t
    posterior_variance = _extract(diffusion.posterior_variance, timesteps, size(x_t))
    posterior_mean, posterior_variance
end

"""
    predict_start_from_noise(diffusion, x_t, timesteps, noise)

Predict an estimate for the ``x_0`` based on the forward process ``q(x_t | x_0)``.
"""
function predict_start_from_noise(diffusion::GaussianDiffusion, x_t::AbstractArray, timesteps::AbstractVector{Int}, noise::AbstractArray)
    coeff1 = _extract(diffusion.sqrt_recip_α_cumprods, timesteps, size(x_t))
    coeff2 = _extract(diffusion.sqrt_recip_α_cumprods_minus_one, timesteps, size(x_t))
    coeff1 .* x_t - coeff2 .* noise
end

function model_predictions(diffusion::GaussianDiffusion, x::AbstractArray, timesteps::AbstractVector{Int})
    noise = diffusion.denoise_fn(x, timesteps)
    x_start = predict_start_from_noise(diffusion, x, timesteps, noise)
    x_start, noise
end

"""
    p_sample(diffusion, x, timesteps, noise; 
        clip_denoised=true, add_noise=true)

The reverse process ``p(x_{t-1} | x_t, t)``. Denoise the data by one timestep.
"""
function p_sample(
    diffusion::GaussianDiffusion, x::AbstractArray, timesteps::AbstractVector{Int}, noise::AbstractArray; 
    clip_denoised::Bool=true, add_noise::Bool=true
    )
    x_start, pred_noise = model_predictions(diffusion, x, timesteps)
    if clip_denoised
        clamp!(x_start, -1, 1)
    end
    posterior_mean, posterior_variance = q_posterior_mean_variance(diffusion, x_start, x, timesteps)
    x_prev = posterior_mean
    if add_noise
        x_prev += sqrt.(posterior_variance) .* noise
    end
    x_prev, x_start
end

"""
    p_sample_loop(diffusion, shape; clip_denoised=true, to_device=cpu)
    p_sample_loop(diffusion, batch_size; options...)

Generate new samples and denoise it to the first time step.
See `p_sample_loop_all` for a version which returns values for all timesteps.
"""
function p_sample_loop(diffusion::GaussianDiffusion, shape::NTuple; clip_denoised::Bool=true, to_device=cpu)
    T = eltype(eltype(diffusion))
    x = randn(T, shape) |> to_device
    @showprogress "Sampling..." for i in diffusion.num_timesteps:-1:1
        timesteps = fill(i, shape[end]) |> to_device;
        noise =  randn(T, size(x)) |> to_device
        x, x_start = p_sample(diffusion, x, timesteps, noise; clip_denoised=clip_denoised, add_noise=(i != 1))
    end
    x
end

function p_sample_loop(diffusion::GaussianDiffusion, batch_size::Int; options...)
    p_sample_loop(diffusion, (diffusion.data_shape..., batch_size); options...)
end

"""
    ddim_sample(diffusion, x, timesteps, timesteps_next, noise; 
        η=1, clip_denoised=true)

Generate new samples using the Denoising Diffusion Implicit Models (DDIM) algorithm proposed.

Reference: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) by Song, Jiaming and Meng, Chenlin and Ermon, Stefano (2020).
"""
function ddim_sample(
        diffusion::GaussianDiffusion, x::AbstractArray, timesteps::AbstractVector{Int}, timesteps_next::AbstractVector{Int},
        noise::AbstractArray; 
        clip_denoised::Bool=true, η::Float32=1.0f0, add_noise::Bool=true
    )
    x_start, pred_noise = model_predictions(diffusion, x, timesteps)
    if clip_denoised
        clamp!(x_start, -1, 1)
    end
    α_cumprod = _extract(diffusion.α_cumprods, timesteps, size(x_start))
    α_cumprod_next = _extract(diffusion.α_cumprods, timesteps_next, size(x_start))
    T = eltype(eltype(diffusion))
    η0 = convert(T, η)
    σ = η0 .* sqrt.((1 .- α_cumprod ./ α_cumprod_next) .* (1 .- α_cumprod_next) ./ (1 .- α_cumprod))
    c = sqrt.(1 .- α_cumprod_next - σ .^ 2)
    x_prev = x_start .* sqrt.(α_cumprod_next) + c .* pred_noise 
    if add_noise
        x_prev += σ .* noise
    end
    x_prev, x_start
end

"""
    ddim_sample_loop(diffusion, sampling_timesteps, shape; η=1; 
        clip_denoised=true, to_device=cpu)
    ddim_sample_loop(diffusion, sampling_timesteps, batch_size; 
        options...)

Generate new samples and denoise it to the first time step using the DDIM algorithm.
Because `sampling_timesteps ≤ diffusion.num_timesteps` this is faster than the standard `p_sample_loop`.

Reference: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) by Song, Jiaming and Meng, Chenlin and Ermon, Stefano (2020)
"""
function ddim_sample_loop(
        diffusion::GaussianDiffusion, sampling_timesteps::Int, shape::NTuple;
        η::AbstractFloat=1.0f0, clip_denoised::Bool=true, to_device=cpu, 
    )

    if  sampling_timesteps > diffusion.num_timesteps
        throw(ErrorException("Require sampling_timesteps ≤ num_timesteps but $sampling_timesteps > $(diffusion.num_timesteps)"))
    end

    T = eltype(eltype(diffusion))
    x = randn(T, shape) |> to_device

    times = reverse(floor.(Int, range(1, diffusion.num_timesteps, length=sampling_timesteps + 1)))
    time_pairs = collect(zip(times[1:end-1], times[2:end]))

    @showprogress "DDIM Sampling..." for (t, t_next) in time_pairs
        timesteps = fill(t, shape[end]) |> to_device;
        timesteps_next = fill(t_next, shape[end]) |> to_device;
        noise = randn(T, size(x)) |> to_device
        x, x_start = ddim_sample(diffusion, x, timesteps, timesteps_next, noise; 
            clip_denoised=clip_denoised, add_noise=(t_next != 1)
        )
    end
    x
end

function ddim_sample_loop(diffusion::GaussianDiffusion, sampling_timesteps::Int, batch_size::Int; options...)
    ddim_sample_loop(diffusion, sampling_timesteps, (diffusion.data_shape..., batch_size); options...)
end

"""
    p_sample_loop_all(diffusion, shape; clip_denoised=true, to_device=cpu)
    p_sample_loop_all(diffusion, batch_size; options...)

Generate new samples and denoise them to the first time step. Return all samples where the last dimension is time.
See `p_sample_loop` for a version which returns only the final sample.
"""
function p_sample_loop_all(diffusion::GaussianDiffusion, shape::NTuple; clip_denoised::Bool=true, to_device=cpu)
    T = eltype(eltype(diffusion))
    x = randn(T, shape) |> to_device
    x_all = Array{T}(undef, size(x)..., 0) |> to_device
    x_start_all = Array{T}(undef, size(x)..., 0) |> to_device
    tdim = ndims(x_all)
    @showprogress "Sampling..." for i in diffusion.num_timesteps:-1:1
        timesteps = fill(i, shape[end]) |> to_device;
        noise =  randn(T, size(x)) |> to_device
        x, x_start = p_sample(diffusion, x, timesteps, noise; clip_denoised=clip_denoised, add_noise=(i != 1))
        x_all = cat(x_all, x, dims=tdim)
        x_start_all = cat(x_start_all, x_start, dims=tdim)
    end
    x_all, x_start_all
end

function p_sample_loop_all(diffusion::GaussianDiffusion, batch_size::Int=16; options...)
    p_sample_loop_all(diffusion, (diffusion.data_shape..., batch_size); options...)
end

"""
    p_losses(diffusion, loss, x_start, timesteps, noise)
    p_losses(diffusion, loss, x_start; to_device=cpu)

Sample from ``q(x_t | x_0)`` and return the loss for the predicted noise.
"""
function p_losses(diffusion::GaussianDiffusion, loss, x_start::AbstractArray, timesteps::AbstractVector{Int}, noise::AbstractArray)
    x = q_sample(diffusion, x_start, timesteps, noise)
    model_out = diffusion.denoise_fn(x, timesteps)

    loss(model_out, noise)
end

function p_losses(diffusion::GaussianDiffusion, loss, x_start::AbstractArray{T, N}; to_device=cpu) where {T, N}
    timesteps = rand(1:diffusion.num_timesteps, size(x_start, N)) |> to_device
    noise = randn(eltype(eltype(diffusion)), size(x_start)) |> to_device
    p_losses(diffusion, loss, x_start, timesteps, noise)
end
