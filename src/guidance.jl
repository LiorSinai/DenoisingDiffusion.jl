"""
    p_losses(diffusion, loss, x_start, timesteps, labels, noise)
    p_losses(diffusion, loss, x_start, labels; to_device=cpu)
    p_losses(diffusion, loss, x_start, embeddings; to_device=cpu)

Sample from ``q(x_t | x_0, c)`` and return the loss for the predicted noise.

Reference: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) by Jonathan Ho, Tim Salimans (2022)  
"""
function p_losses(
    diffusion::GaussianDiffusion,
    loss,
    x_start::AbstractArray{T,N},
    timesteps::AbstractVector{Int},
    labels_or_embeddings::AbstractVecOrMat,
    noise::AbstractArray
    ) where {T,N}
    x = q_sample(diffusion, x_start, timesteps, noise)
    model_out = diffusion.denoise_fn(x, timesteps, labels_or_embeddings)
    loss(model_out, noise)
end

function p_losses(
    diffusion::GaussianDiffusion,
    loss,
    x_start::AbstractArray,
    labels_or_embeddings::AbstractVecOrMat,
    ; to_device=cpu
    )
    batch_size = size(x_start)[end]
    label_batch_size = size(labels_or_embeddings)[end]
    @assert(batch_size == label_batch_size,
        "batch size != label length, $batch_size != $(label_batch_size)"
    )
    timesteps = rand(1:diffusion.num_timesteps, batch_size) |> to_device
    noise = randn(eltype(eltype(diffusion)), size(x_start)) |> to_device
    p_losses(diffusion, loss, x_start, timesteps, labels_or_embeddings, noise)
end

"""
    p_sample_loop(diffusion, shape, labels_or_embeddings; 
        clip_denoised=true, to_device=cpu, guidance_scale=1.0f0)
    p_sample_loop(diffusion, labels_or_embeddings; options...)
    p_sample_loop(diffusion, batch_size, label; options...)

Generate new samples and denoise it to the first time step using the classifier free guidance algorithm.
See `p_sample_loop_all` for a version which returns values for all timesteps.

Reference: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) by Jonathan Ho, Tim Salimans (2022)  
"""
function p_sample_loop(
    diffusion::GaussianDiffusion, shape::NTuple, labels_or_embeddings::AbstractVecOrMat
    ; clip_denoised::Bool=true, to_device=cpu, guidance_scale::AbstractFloat=1.0f0
    )
    T = eltype(eltype(diffusion))
    x = randn(T, shape) |> to_device
    @showprogress "Sampling ..." for i in diffusion.num_timesteps:-1:1
        timesteps = fill(i, shape[end]) |> to_device
        noise = randn(T, size(x)) |> to_device
        x, x_start = p_sample(
            diffusion, x, timesteps, labels_or_embeddings, noise
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

function p_sample_loop(diffusion::GaussianDiffusion, embeddings::AbstractMatrix; options...)
    batch_size = size(embeddings, 2)
    p_sample_loop(diffusion, (diffusion.data_shape..., batch_size), embeddings; options...)
end

"""
    p_sample_loop_all(diffusion, shape, labels_or_embeddings; 
        clip_denoised=true, to_device=cpu, guidance_scale=1.0f0)
    p_sample_loop_all(diffusion, labels; options...)
    p_sample_loop_all(diffusion, batch_size, label; options...)

Generate new samples and denoise them to the first time step. Return all samples where the last dimension is time.
See `p_sample_loop` for a version which returns only the final sample.

Reference: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) by Jonathan Ho, Tim Salimans (2022)  
"""
function p_sample_loop_all(
    diffusion::GaussianDiffusion, shape::NTuple, labels_or_embeddings::AbstractVecOrMat
    ; clip_denoised::Bool=true, to_device=cpu, guidance_scale::AbstractFloat=1.0f0
)
    T = eltype(eltype(diffusion))
    x = randn(T, shape) |> to_device
    x_all = Array{T}(undef, size(x)..., 0) |> to_device
    x_start_all = Array{T}(undef, size(x)..., 0) |> to_device
    dim_time = ndims(x_all)
    @showprogress "Sampling..." for i in diffusion.num_timesteps:-1:1
        timesteps = fill(i, shape[end]) |> to_device
        noise = randn(T, size(x)) |> to_device
        x, x_start = p_sample(
            diffusion, x, timesteps, labels_or_embeddings, noise
            ; clip_denoised=clip_denoised, add_noise=(i != 1), guidance_scale=guidance_scale
        )
        x_all = cat(x_all, x, dims=dim_time)
        x_start_all = cat(x_start_all, x_start, dims=dim_time)
    end
    x_all, x_start_all
end

function p_sample_loop_all(diffusion::GaussianDiffusion, batch_size::Int, label::Int; options...)
    labels = fill(label, batch_size)
    shape = (diffusion.data_shape..., batch_size)
    p_sample_loop_all(diffusion, shape, labels; options...)
end

function p_sample_loop_all(diffusion::GaussianDiffusion, labels::AbstractVector{Int}; options...)
    batch_size = length(labels)
    shape = (diffusion.data_shape..., batch_size)
    p_sample_loop_all(diffusion, shape, labels; options...)
end

function p_sample_loop_all(diffusion::GaussianDiffusion, embeddings::AbstractVecOrMat; options...)
    batch_size = size(embeddings, 2)
    shape = (diffusion.data_shape..., batch_size)
    p_sample_loop_all(diffusion, shape, embeddings; options...)
end

"""
    p_sample(diffusion, x, timesteps, labels_or_embeddings, noise; 
        clip_denoised=true, add_noise::Bool=true, guidance_scale=1.0f0)

The reverse process ``p(x_{t-1} | x_t, t, c)``. Denoise the data by one timestep conditioned on labels.

Reference: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) by Jonathan Ho, Tim Salimans (2022) 
"""
function p_sample(
    diffusion::GaussianDiffusion,
    x::AbstractArray,
    timesteps::AbstractVector{Int},
    labels_or_embeddings::AbstractVecOrMat,
    noise::AbstractArray
    ;
    clip_denoised::Bool=true,
    add_noise::Bool=true,
    guidance_scale::AbstractFloat=1.0f0
    )
    if guidance_scale == 1.0f0
        x_start, pred_noise = denoise(diffusion, x, timesteps, labels_or_embeddings)
    else
        x_start, pred_noise = classifier_free_guidance(
            diffusion, x, timesteps, labels_or_embeddings; guidance_scale=guidance_scale
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

function denoise(
    diffusion::GaussianDiffusion,
    x::AbstractArray,
    timesteps::AbstractVector{Int},
    labels_or_embeddings::AbstractVecOrMat
    )
    noise = diffusion.denoise_fn(x, timesteps, labels_or_embeddings)
    x_start = predict_start_from_noise(diffusion, x, timesteps, noise)
    x_start, noise
end

function classifier_free_guidance(
    diffusion::GaussianDiffusion,
    x::AbstractArray,
    timesteps::AbstractVector{Int},
    labels::AbstractVector{Int}
    ; guidance_scale=1.0f0
    )
    T = eltype(eltype(diffusion))
    guidance_scale_ = convert(T, guidance_scale)
    batch_size = size(x)[end]
    x_double = cat(x, x, dims=ndims(x))
    timesteps_double = vcat(timesteps, timesteps)
    labels_both = vcat(labels, fill(1, batch_size))

    noise_both = diffusion.denoise_fn(x_double, timesteps_double, labels_both)

    inds = ntuple(Returns(:), ndims(x_double) - 1)
    ϵ_cond = view(noise_both, inds..., 1:batch_size)
    ϵ_uncond = view(noise_both, inds..., (batch_size+1):(2*batch_size))
    noise = ϵ_uncond + guidance_scale_ * (ϵ_cond - ϵ_uncond)

    x_start = predict_start_from_noise(diffusion, x, timesteps, noise)
    x_start, noise
end

function classifier_free_guidance(
    diffusion::GaussianDiffusion,
    x::AbstractArray,
    timesteps::AbstractVector{Int},
    embeddings::AbstractMatrix
    ; guidance_scale=1.0f0
    )
    T = eltype(eltype(diffusion))
    guidance_scale_ = convert(T, guidance_scale)
    batch_size = size(x)[end]

    x_double = cat(x, x, dims=ndims(x))
    timesteps_double = vcat(timesteps, timesteps)
    unconditioned_embeddings = zeros(size(embeddings)...)
    embeddings_both = hcat(embeddings, unconditioned_embeddings)
    noise_both = diffusion.denoise_fn(x_double, timesteps_double, embeddings_both)

    inds = ntuple(Returns(:), ndims(x_double) - 1)
    ϵ_cond = view(noise_both, inds..., 1:batch_size)
    ϵ_uncond = view(noise_both, inds..., (batch_size+1):(2*batch_size))
    noise = ϵ_uncond + guidance_scale_ * (ϵ_cond - ϵ_uncond)

    x_start = predict_start_from_noise(diffusion, x, timesteps, noise)
    x_start, noise
end
