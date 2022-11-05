"""
    p_losses(diffusion, loss, x_start, timesteps, labels, noise)
    p_losses(diffusion, loss, xy; to_device=cpu, p_uncond=0.2)

Sample from ``q(x_t | x_0, c)`` and return the loss for the predicted noise.

Reference: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) by Jonathan Ho, Tim Salimans (2022)  
"""
function p_losses(
        diffusion::GaussianDiffusion, loss, x_start::AbstractArray{T, N}, 
        timesteps::AbstractVector{Int}, labels::AbstractVector{Int}, noise::AbstractArray
    ) where {T, N}
    if (size(x_start, N) != length(labels))
        throw(DimensionMismatch("batch size != label length, $N != $(length(labels))"))
    end
    x = q_sample(diffusion, x_start, timesteps, noise)
    model_out = diffusion.denoise_fn(x, timesteps, labels)
    loss(model_out, noise)
end

function p_losses(
        diffusion::GaussianDiffusion, loss, xy::Tuple{AbstractArray, AbstractVector}; 
        to_device=cpu, p_uncond::Float64=0.20
    )
    x_start = xy[1]
    labels = xy[2]
    batch_size = size(x_start)[end]
    if (batch_size != length(labels))
        throw(DimensionMismatch("batch size != label length, $batch_size != $(length(labels))"))
    end
    timesteps = rand(1:diffusion.num_timesteps, batch_size ) |> to_device
    noise = randn(eltype(eltype(diffusion)), size(x_start)) |> to_device
    # with probability p_uncond we train without class conditioning
    labels = labels |> cpu
    is_class_cond = rand(batch_size) .>= p_uncond
    is_not_class_cond = .~is_class_cond
    labels = (labels .* is_class_cond) + is_not_class_cond # set is_not_class_cond to 1
    labels = labels |> to_device
    p_losses(diffusion, loss, x_start, timesteps, labels, noise)
end

"""
    p_sample_loop(diffusion, shape, labels; 
        clip_denoised=true, to_device=cpu, guidance_scale=1.0f0)
    p_sample_loop(diffusion, labels; options...)
    p_sample_loop(diffusion, batch_size, label; options...)

Generate new samples and denoise it to the first time step using the classifier free guidance algorithm.
See `p_sample_loop_all` for a version which returns values for all timesteps.

Reference: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) by Jonathan Ho, Tim Salimans (2022)  
"""
function p_sample_loop(
        diffusion::GaussianDiffusion, shape::NTuple, labels::AbstractVector{Int}
        ; clip_denoised::Bool=true, to_device=cpu, guidance_scale::AbstractFloat=1.0f0
    )
    T = eltype(eltype(diffusion))
    x = randn(T, shape) |> to_device
    @showprogress "Sampling ..." for i in diffusion.num_timesteps:-1:1
        timesteps = fill(i, shape[end]) |> to_device;
        noise =  randn(T, size(x)) |> to_device
        x, x_start = p_sample(
                diffusion, x, timesteps, labels, noise
                ; clip_denoised=clip_denoised, add_noise=(i != 1), guidance_scale=guidance_scale
            )
    end
    x
end

function p_sample_loop(diffusion::GaussianDiffusion, batch_size::Int, label::Int; options...)
    labels = fill(label, batch_size)
    p_sample_loop(diffusion, (diffusion.data_shape..., batch_size), labels; options...)
end

function p_sample_loop(diffusion::GaussianDiffusion, labels::AbstractVector{Int}; options...)
    batch_size = length(labels)
    p_sample_loop(diffusion, (diffusion.data_shape..., batch_size), labels; options...)
end

"""
    ddim_sample_loop(diffusion, sampling_timesteps, shape, labels; 
        η=1, clip_denoised=true, to_device=cpu, guidance_scale=1.0f0)
    ddim_sample_loop(diffusion, sampling_timesteps, batch_size, label; options...)
    ddim_sample_loop(diffusion, sampling_timesteps, labels; options...)

Generate new samples and denoise it to the first time step using the DDIM algorithm combined with the classifier-free guidance algorithm.
Because `sampling_timesteps ≤ diffusion.num_timesteps` this is faster than the standard `p_sample_loop`.

References: 
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) by Song, Jiaming and Meng, Chenlin and Ermon, Stefano (2020)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) by Jonathan Ho, Tim Salimans (2022) 
""" 
function ddim_sample_loop(
    diffusion::GaussianDiffusion, sampling_timesteps::Int, shape::NTuple, labels::AbstractVector{Int};
    η::AbstractFloat=1.0f0, clip_denoised::Bool=true, to_device=cpu, guidance_scale::AbstractFloat=1.0f0,
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
        x, x_start = ddim_sample(diffusion, x, timesteps, timesteps_next, labels, noise; 
            clip_denoised=clip_denoised, add_noise=(t_next != 1), η=η, guidance_scale=guidance_scale
        )
    end
    x
end

function ddim_sample_loop(diffusion::GaussianDiffusion, sampling_timesteps::Int, labels::AbstractVector{Int}; options...)
    batch_size = length(labels)
    ddim_sample_loop(diffusion, sampling_timesteps, (diffusion.data_shape..., batch_size), labels; options...)
end

function ddim_sample_loop(diffusion::GaussianDiffusion, sampling_timesteps::Int, batch_size::Int, label::Int; options...)
    labels = fill(label, batch_size)
    ddim_sample_loop(diffusion, sampling_timesteps, (diffusion.data_shape..., batch_size), labels; options...)
end

"""
    p_sample_loop_all(diffusion, shape, labels; 
        clip_denoised=true, to_device=cpu, guidance_scale=1.0f0)
    p_sample_loop_all(diffusion, labels; options...)
    p_sample_loop_all(diffusion, batch_size, label; options...)

Generate new samples and denoise them to the first time step. Return all samples where the last dimension is time.
See `p_sample_loop` for a version which returns only the final sample.

Reference: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) by Jonathan Ho, Tim Salimans (2022)  
"""
function p_sample_loop_all(
        diffusion::GaussianDiffusion, shape::NTuple, labels::AbstractVector{Int}
        ; clip_denoised::Bool=true, to_device=cpu, guidance_scale::AbstractFloat=1.0f0
    )
    T = eltype(eltype(diffusion))
    x = randn(T, shape) |> to_device
    x_all = Array{T}(undef, size(x)..., 0) |> to_device
    x_start_all = Array{T}(undef, size(x)..., 0) |> to_device
    dim_time = ndims(x_all)
    @showprogress "Sampling..." for i in diffusion.num_timesteps:-1:1
        timesteps = fill(i, shape[end]) |> to_device;
        noise =  randn(T, size(x)) |> to_device
        x, x_start = p_sample(
                diffusion, x, timesteps, labels, noise
                ; clip_denoised=clip_denoised, add_noise=(i != 1), guidance_scale=guidance_scale
            )
        x_all = cat(x_all, x, dims=dim_time)
        x_start_all = cat(x_start_all, x_start, dims=dim_time)
    end
    x_all, x_start_all
end

function p_sample_loop_all(diffusion::GaussianDiffusion, batch_size::Int, label::Int; options...)
    labels = fill(label, batch_size)
    p_sample_loop_all(diffusion, (diffusion.data_shape..., batch_size), labels; options...)
end

function p_sample_loop_all(diffusion::GaussianDiffusion, labels::AbstractVector{Int}; options...)
    batch_size = length(labels)
    p_sample_loop_all(diffusion, (diffusion.data_shape..., batch_size), labels; options...)
end

"""
    p_sample(diffusion, x, timesteps, labels, noise; 
        clip_denoised=true, add_noise::Bool=true, guidance_scale=1.0f0)

The reverse process ``p(x_{t-1} | x_t, t, c)``. Denoise the data by one timestep conditioned on labels.

Reference: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) by Jonathan Ho, Tim Salimans (2022) 
"""
function p_sample(
        diffusion::GaussianDiffusion, x::AbstractArray, timesteps::AbstractVector{Int}, labels::AbstractVector{Int}, noise::AbstractArray; 
        clip_denoised::Bool=true, add_noise::Bool=true, guidance_scale::AbstractFloat=1.0f0
    )
    if guidance_scale == 1.0f0
        x_start, pred_noise = model_predictions(diffusion, x, timesteps, labels)
    else
        x_start, pred_noise = _classifier_free_guidance(
            diffusion, x, timesteps, labels; guidance_scale=guidance_scale
        )
    end
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
    ddim_sample(diffusion, x, timesteps, timesteps_next, labels, noise;
        η=1, clip_denoised=true, guidance_scale=1.0f)

Generate new samples using the DDIM algorithm combined with the classifier-free guidance algorithm.

References: 
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) by Song, Jiaming and Meng, Chenlin and Ermon, Stefano (2020)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) by Jonathan Ho, Tim Salimans (2022) 
"""
function ddim_sample(
    diffusion::GaussianDiffusion, x::AbstractArray, timesteps::AbstractVector{Int}, timesteps_next::AbstractVector{Int},
    labels::AbstractVector{Int}, noise::AbstractArray; 
    clip_denoised::Bool=true, η::AbstractFloat=1.0f0, add_noise::Bool=true, guidance_scale::AbstractFloat=1.0f0
    )
    if guidance_scale == 1.0f0
        x_start, pred_noise = model_predictions(diffusion, x, timesteps, labels)
    else
        x_start, pred_noise = _classifier_free_guidance(
            diffusion, x, timesteps, labels; guidance_scale=guidance_scale
        )
    end
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

function model_predictions(
    diffusion::GaussianDiffusion, 
    x::AbstractArray, timesteps::AbstractVector{Int}, labels::AbstractVector{Int}
)
    noise = diffusion.denoise_fn(x, timesteps, labels)
    x_start = predict_start_from_noise(diffusion, x, timesteps, noise)
    x_start, noise
end

function _classifier_free_guidance(
        diffusion::GaussianDiffusion, 
        x::AbstractArray, timesteps::AbstractVector{Int}, labels::AbstractVector{Int}
        ; guidance_scale=1.0f0
    )
    T = eltype(eltype(diffusion))
    guidance_scale_ = convert(T, guidance_scale)
    batch_size = size(x)[end]
    x_double = cat(x, x, dims=ndims(x))
    timesteps_double = vcat(timesteps, timesteps)
    labels_both = vcat(labels, fill(1, batch_size))

    noise_both = diffusion.denoise_fn(x_double, timesteps_double, labels_both)

    inds = ntuple(Returns(:), ndims(x_double)-1)
    ϵ_cond = view(noise_both, inds..., 1:batch_size)
    ϵ_uncond = view(noise_both, inds..., (batch_size + 1):(2 * batch_size))
    noise = ϵ_uncond + guidance_scale_ * (ϵ_cond - ϵ_uncond)

    x_start = predict_start_from_noise(diffusion, x, timesteps, noise)
    x_start, noise
end
