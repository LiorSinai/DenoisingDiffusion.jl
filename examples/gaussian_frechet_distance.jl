"""
    gaussian_frechet_distance(μ1, Σ1, μ2, Σ2)

The Frechet distance between two multivariate Gaussians ``X_1 ~ N(μ1, Σ1)``
and ``X_2 ~ N(μ2, Σ2)`` is

``d^2 = ||μ1 - μ2||^2 + tr(Σ1 + Σ2 - 2*\\sqrt{Σ1*Σ2})``
"""
function gaussian_frechet_distance(μ1::AbstractMatrix, Σ1::AbstractMatrix, μ2::AbstractMatrix, Σ2::AbstractMatrix)
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
