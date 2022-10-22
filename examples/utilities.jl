using StatsBase
using LinearAlgebra

## model sizes 

function conv_output_size(input_size::Int, filter_size::Int, stride::Int=1, pad::Int=0)
    floor(Int, (input_size + 2 * pad - filter_size)/stride) + 1
end

count_parameters(model) = sum(length, Flux.params(model))

### normalize

function normalize_zero_to_one(x)
    x_min, x_max = extrema(x)
    x_norm = (x .- x_min) ./ (x_max - x_min)
    x_norm
end

function normalize_neg_one_to_one(x)
    2 * normalize_zero_to_one(x) .- 1
end

unnormalize_zero_to_one(x::AbstractArray) = (x .+ one(x[1])) / 2 ;

"""
    img_WHC_to_rgb(img_WHC) where {T}

Converts images in (width, height, channel) form to RGB images.
Should normalize the images first for better constrasts.
"""
function img_WHC_to_rgb(img_WHC::AbstractArray{T,N}) where {T,N}
    @assert N == 3 || N == 4
    @assert size(img_WHC, 3) == 3
    img_CHW = permutedims(img_WHC, (3, 2, 1, 4:N...))
    img = colorview(RGB, img_CHW)
    img
end

### evaluation functions

"""
    argmin_func(f, df, ddf, tmin, tmax; num_iters=10, length=100)

Find the minimum of a function `f` in the range `tmin:tmax`.
"""
function argmin_func(f, df, ddf, tmin::AbstractFloat, tmax::AbstractFloat; num_iters::Int=10, length::Int=100)
    seed = argmin(f, range(tmin, tmax, length))
    root = newtons_method(df, ddf, seed, tmin, tmax; num_iters=num_iters)
    root
end

"""
    newtons_method(f, fgrad, root, rmin, rmax; num_iters=10)

Uses Newton's method to find the root for the equation `f(root)=0`. 
If a function has multiple roots a good initial `root` is required to get an answer in the desired region.
"""
function newtons_method(f, fgrad, root::AbstractFloat, rmin::AbstractFloat, rmax::AbstractFloat; num_iters::Int=10)
    for i in 1:num_iters
        root = root - f(root)/fgrad(root)
        root = clamp(root, rmin, rmax)
    end
    root
end

"""
    gaussian_fretchet_distance(μ1, Σ1, μ2, Σ2)

The Frechet distance between two multivariate Gaussians X_1 ~ N(μ1, Σ1)
and X_2 ~ N(μ2, Σ2) is
    d^2 = ||μ1 - μ2||^2 + tr(Σ1 + Σ2 - 2*sqrt(Σ1*Σ2))
"""
function gaussian_fretchet_distance(μ1::AbstractMatrix, Σ1::AbstractMatrix, μ2::AbstractMatrix, Σ2::AbstractMatrix)
    diff = μ1 - μ2
    covmean = sqrt(Σ1 * Σ2)
    if eltype(covmean) <: Complex
        @warn("sqrt(Σ1 * Σ2) is complex")
        if all(isapprox.(0.0, diag(imag(covmean)), atol=1e-3))
            @info("imaginary components are small and have been set to zero")
            covmean = real(covmean)
        end
    end
    sum(diff .* diff) + tr(Σ1 + Σ2 - 2 * covmean)
end

function get_activation_statistics(activations::AbstractArray)
    μ = mean(activations; dims=2)
    Σ = cov(activations; dims=2, corrected=true)
    μ, Σ
end
