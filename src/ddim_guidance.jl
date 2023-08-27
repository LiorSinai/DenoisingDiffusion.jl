"""
    ddim_sample(diffusion, x, timesteps, timesteps_next, labels, noise;
        η=1, clip_denoised=true, guidance_scale=1.0f)

Generate new samples using the DDIM algorithm combined with the classifier-free guidance algorithm.

References: 
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) by Song, Jiaming and Meng, Chenlin and Ermon, Stefano (2020)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) by Jonathan Ho, Tim Salimans (2022) 
"""
function ddim_sample(
    diffusion::GaussianDiffusion,
    x::AbstractArray,
    timesteps::AbstractVector{Int},
    timesteps_next::AbstractVector{Int},
    labels::AbstractVector{Int},
    noise::AbstractArray
    ;
    clip_denoised::Bool=true,
    η::AbstractFloat=1.0f0,
    add_noise::Bool=true,
    guidance_scale::AbstractFloat=1.0f0
    )
    if guidance_scale == 1.0f0
        x_start, pred_noise = denoise(diffusion, x, timesteps, labels)
    else
        x_start, pred_noise = classifier_free_guidance(
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
    diffusion::GaussianDiffusion,
    sampling_timesteps::Int,
    shape::NTuple,
    labels::AbstractVector{Int}
    ;
    η::AbstractFloat=1.0f0,
    clip_denoised::Bool=true,
    to_device=cpu,
    guidance_scale::AbstractFloat=1.0f0
    )
    if sampling_timesteps > diffusion.num_timesteps
        throw(ErrorException("Require sampling_timesteps ≤ num_timesteps but $sampling_timesteps > $(diffusion.num_timesteps)"))
    end

    T = eltype(eltype(diffusion))
    x = randn(T, shape) |> to_device

    times = reverse(floor.(Int, range(1, diffusion.num_timesteps, length=sampling_timesteps + 1)))
    time_pairs = collect(zip(times[1:end-1], times[2:end]))

    @showprogress "DDIM Sampling..." for (t, t_next) in time_pairs
        timesteps = fill(t, shape[end]) |> to_device
        timesteps_next = fill(t_next, shape[end]) |> to_device
        noise = randn(T, size(x)) |> to_device
        x, x_start = ddim_sample(diffusion, x, timesteps, timesteps_next, labels, noise;
            clip_denoised=clip_denoised, add_noise=(t_next != 1), η=η, guidance_scale=guidance_scale
        )
    end
    x
end

function ddim_sample_loop(diffusion::GaussianDiffusion, sampling_timesteps::Int, labels::AbstractVector{Int}; options...)
    batch_size = length(labels)
    shape = (diffusion.data_shape..., batch_size)
    ddim_sample_loop(diffusion, sampling_timesteps, shape, labels; options...)
end

function ddim_sample_loop(diffusion::GaussianDiffusion, sampling_timesteps::Int, batch_size::Int, label::Int; options...)
    labels = fill(label, batch_size)
    shape = (diffusion.data_shape..., batch_size)
    ddim_sample_loop(diffusion, sampling_timesteps, shape, labels; options...)
end