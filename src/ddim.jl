"""
    ddim_sample(diffusion, x, timesteps, timesteps_next, noise; 
        η=1, clip_denoised=true)

Generate new samples using the Denoising Diffusion Implicit Models (DDIM) algorithm.

Reference: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) by Song, Jiaming and Meng, Chenlin and Ermon, Stefano (2020).
"""
function ddim_sample(
    diffusion::GaussianDiffusion,
    x::AbstractArray,
    timesteps::AbstractVector{Int}, 
    timesteps_next::AbstractVector{Int},
    noise::AbstractArray
    ;
    clip_denoised::Bool=true,
    η::Float32=1.0f0,
    add_noise::Bool=true
    )
    x_start, pred_noise = denoise(diffusion, x, timesteps)
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
    diffusion::GaussianDiffusion,
    sampling_timesteps::Int,
    shape::NTuple
    ;
    η::AbstractFloat=1.0f0,
    clip_denoised::Bool=true,
    to_device=cpu
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
        x, x_start = ddim_sample(diffusion, x, timesteps, timesteps_next, noise;
            clip_denoised=clip_denoised, add_noise=(t_next != 1), η=η
        )
    end
    x
end

function ddim_sample_loop(diffusion::GaussianDiffusion, sampling_timesteps::Int, batch_size::Int; options...)
    shape = (diffusion.data_shape..., batch_size)
    ddim_sample_loop(diffusion, sampling_timesteps, shape; options...)
end
