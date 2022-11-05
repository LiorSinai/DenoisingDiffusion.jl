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
    argmin_func_newton(f, df, ddf, tmin, tmax; num_iters=10, length=100)

Find the minimum of a function `f` in the range `tmin:tmax` using Newton's method.
"""
function argmin_func_newton(f, df, ddf, tmin::AbstractFloat, tmax::AbstractFloat; num_iters::Int=10, length::Int=100)
    seed = argmin(f, range(tmin, tmax, length))
    root = newtons_method(df, ddf, seed, tmin, tmax; num_iters=num_iters)
    root
end

"""
    argmin_func_bisection(f, df, ddf, tmin, tmax; num_iters=10, length=100)

Find the minimum of a function `f` in the range `tmin:tmax` using the Bisection method.
"""
function argmin_func_bisection(f, df, tmin::AbstractFloat, tmax::AbstractFloat; num_iters::Int=10, length::Int=100)
    seed = argmin(f, range(tmin, tmax, length))
    Δ = 0.5 * (tmax - tmin) / length
    left = max(seed - Δ, tmin)
    right = min(seed + Δ, tmax)
    root = bisection_method(df, left, right; num_iters=num_iters)
    root
end

"""
    newtons_method(f, fgrad, root, rmin, rmax; num_iters=10, ϵ=0.5)

Uses Newton's method to find the root for the equation `f(root)=0`. 
If a function has multiple roots a good initial `root` is required to get an answer in the desired region.
"""
function newtons_method(f, fgrad, root::AbstractFloat, rmin::AbstractFloat, rmax::AbstractFloat; num_iters::Int=10, ϵ::AbstractFloat=0.3)
    grad0 = fgrad(root)
    if (abs(grad0) < ϵ) 
        #@warn("gradient=$grad0 is too low for Newton's method. Returning root without optimization.")
        return root
    end
    for i in 1:num_iters
        root = root - f(root)/fgrad(root)
        root = clamp(root, rmin, rmax)
    end
    root
end

"""
    bisection_method(f, left, right; num_iters=10).
"""
function bisection_method(f, left::AbstractFloat, right::AbstractFloat; num_iters::Int=10)
    if sign(f(left)) == sign(f(right))
        #@warn("sign(f(left)) == sign(f(right)). Returning middle without optimization.")
        return (left + right)/2
    end
    for i in 1:num_iters
        middle = (left + right)/2
        if sign(f(middle)) == sign(f(left))
            left = middle
        else
            right = middle
        end
    end
    (left + right) /2
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
