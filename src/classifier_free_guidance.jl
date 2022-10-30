"""
    p_lossess_guided(diffusion, loss, x_start, timesteps,  labels, noise)
    p_lossess_guided(diffusion, loss, xy; to_device=cpu, p_uncond=0.2)

Sample from `q(x_t | x_0, labels)` and return the loss for the predicted noise.
"""
function p_lossess_guided(diffusion::GaussianDiffusion, loss, x_start::AbstractArray{T, N}, 
        timesteps::AbstractVector{Int}, labels::AbstractVector{Int}, noise::AbstractArray) where {T, N}
    if (size(x_start, N) != length(labels))
        throw(DimensionMismatch("batch size != label length, $N != $(length(labels))"))
    end
    x = q_sample(diffusion, x_start, timesteps, noise)
    model_out = diffusion.denoise_fn(x, timesteps, labels)
    loss(model_out, noise)
end

function p_lossess_guided(diffusion::GaussianDiffusion, loss, xy::Tuple; to_device=cpu, p_uncond::Float64=0.20)
    x_start = xy[1]
    labels = xy[2]
    N = size(x_start)[end]
    if (N != length(labels))
        throw(DimensionMismatch("batch size != label length, $N != $(length(labels))"))
    end
    timesteps = rand(1:diffusion.num_timesteps, N ) |> to_device
    noise = randn(eltype(eltype(diffusion)), size(x_start)) |> to_device
    # with probability p_uncond we train without class conditioning
    is_class_cond = rand(N) .>= p_uncond
    is_not_class_cond = .~is_class_cond
    labels = (labels .* is_class_cond) + is_not_class_cond # set is_not_class_cond to 1
    p_lossess_guided(diffusion, loss, x_start, timesteps, labels, noise)
end

"""
    p_sample_loop_guided(diffusion, shape, labels; clip_denoised=true, to_device=cpu, guidance_scale=1.0f0)
    p_sample_loop_guided(diffusion, labels; options...)
    p_sample_loop_guided(diffusion, batch_size, label; options...)

Generate new samples and denoise it to the first time step using the classifier free guidance algorithm.
See `p_sample_loop_all_guided` for a version which returns values for all timesteps.
"""
function p_sample_loop_guided(diffusion::GaussianDiffusion, shape::NTuple, labels::AbstractVector{Int}; 
        clip_denoised::Bool=true, to_device=cpu, guidance_scale::Float32=1.0f0
    )
    T = eltype(eltype(diffusion))
    guidance_scale_ = convert(T, guidance_scale)
    batch_size = shape[end]
    x = randn(T, shape) |> to_device
    y = vcat(labels, fill(1, batch_size))
    @showprogress "Sampling ..." for i in diffusion.num_timesteps:-1:1
        timesteps = fill(i, 2 * batch_size) |> to_device;
        x_double = hcat(x, x)
        noise =  randn(T, size(x_double)) |> to_device
        x_double, x_start = p_sample(diffusion, x_double, timesteps, y, noise; clip_denoised=clip_denoised, add_noise=(i != 1))
        x_cond = view(x_double, :, 1:batch_size)
        x_uncond = view(x_double, :, (batch_size + 1):(2 * batch_size))
        x = x_uncond + guidance_scale_ * (x_cond - x_uncond)
    end
    x
end

function p_sample_loop_guided(diffusion::GaussianDiffusion, batch_size::Int, label::Int; options...)
    labels = fill(1 + label, batch_size)
    p_sample_loop_guided(diffusion, (diffusion.data_shape..., batch_size), labels; options...)
end

function p_sample_loop_guided(diffusion::GaussianDiffusion, labels::AbstractVector{Int}; options...)
    batch_size = length(labels)
    p_sample_loop_guided(diffusion, (diffusion.data_shape..., batch_size), labels; options...)
end

"""
    p_sample_loop_all_guided(diffusion, shape, labels; clip_denoised=true, to_device=cpu, guidance_scale=1.0f0)
    p_sample_loop_all_guided(diffusion, labels; options...)
    p_sample_loop_all_guided(diffusion, batch_size, label; options...)

Generate new samples and denoise them to the first time step. Return all samples where the last dimension is time.
See `p_sample_loop` for a version which returns only the final sample.
"""
function p_sample_loop_all_guided(diffusion::GaussianDiffusion, shape::NTuple, labels::AbstractVector{Int};
    clip_denoised::Bool=true, to_device=cpu, guidance_scale::Float32=1.0f0
    )
    T = eltype(eltype(diffusion))
    guidance_scale_ = convert(T, guidance_scale)
    batch_size = shape[end]
    x = randn(T, shape) |> to_device
    x_all = Array{T}(undef, size(x)..., 0) |> to_device
    x_start_all = Array{T}(undef, size(x)..., 0) |> to_device
    y = vcat(labels, fill(1, batch_size))
    dim_time = length(size(x_all)) 
    @showprogress "Sampling..." for i in diffusion.num_timesteps:-1:1
        timesteps = fill(i, 2 * batch_size) |> to_device;
        x_double = hcat(x, x)
        noise =  randn(T, size(x_double)) |> to_device
        x_double, x_double_start = p_sample(diffusion, x_double, timesteps, y, noise; clip_denoised=clip_denoised, add_noise=(i != 1))
        x_cond = view(x_double, :, 1:batch_size)
        x_uncond = view(x_double, :, (batch_size + 1):(2 * batch_size))
        x = x_uncond + guidance_scale_ * (x_cond - x_uncond)
        x_start = view(x_double_start, :, 1:batch_size)
        x_all = cat(x_all, x, dims=dim_time)
        x_start_all = cat(x_start_all, x_start, dims=dim_time)
    end
    x_all, x_start_all
end

function p_sample_loop_all_guided(diffusion::GaussianDiffusion, batch_size::Int, label::Int; options...)
    labels = fill(1 + label, batch_size)
    p_sample_loop_all_guided(diffusion, (diffusion.data_shape..., batch_size), labels; options...)
end

function p_sample_loop_all_guided(diffusion::GaussianDiffusion, labels::AbstractVector{Int}; options...)
    batch_size = length(labels)
    p_sample_loop_all_guided(diffusion, (diffusion.data_shape..., batch_size), labels; options...)
end

"""
    p_sample(diffusion, x, timesteps, tokens, noise; clip_denoised=true)

The reverse process `p(x_{t-1} | x_t, t, tokens)`. Denoise the data by one timestep conditioned on tokens.
"""
function p_sample(
        diffusion::GaussianDiffusion, x::AbstractArray, timesteps::AbstractVector{Int}, tokens::AbstractVector{Int}, noise::AbstractArray; 
        clip_denoised::Bool=true, add_noise::Bool=true
    )
    x_start, pred_noise = model_predictions(diffusion, x, timesteps, tokens; clip_denoised=clip_denoised)
    posterior_mean, posterior_variance = q_posterior_mean_variance(diffusion, x_start, x, timesteps)
    x_prev = posterior_mean
    if add_noise
        x_prev += sqrt.(posterior_variance) .* noise
    end
    x_prev, x_start
end

function model_predictions(
        diffusion::GaussianDiffusion, 
        x::AbstractArray, timesteps::AbstractVector{Int}, tokens::AbstractVector{Int}
        ; clip_denoised::Bool=true
    )
    noise = diffusion.denoise_fn(x, timesteps, tokens)
    x_start = predict_start_from_noise(diffusion, x, timesteps, noise)

    if clip_denoised
        clamp!(x_start, -1, 1)
    end

    x_start, noise
end
