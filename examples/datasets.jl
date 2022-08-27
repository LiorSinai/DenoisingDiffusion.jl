# based on https://scikit-learn.org/stable/modules/classes.html#samples-generator

using Random

function make_spiral(rng::AbstractRNG, n_samples::Int=1000)
    t_min = 1.5π
    t_max = 4.5π

    t = rand(rng, n_samples) * (t_max - t_min) .+ t_min

    x = t .* cos.(t)
    y = t .* sin.(t)

    permutedims([x y], (2, 1))
end

make_spiral(n_samples::Int=1000) = make_spiral(Random.GLOBAL_RNG, n_samples)

function make_moons(rng::AbstractRNG, n_samples::Int=1000)
    n_moons = floor(Int, n_samples /2)
    t_min = 0.0
    t_max = π
    t_inner = rand(rng, n_moons) * (t_max - t_min) .+ t_min
    t_outer = rand(rng, n_moons) * (t_max - t_min) .+ t_min
    outer_circ_x = cos.(t_outer)
    outer_circ_y = sin.(t_outer)
    inner_circ_x = 1 .- cos.(t_inner)
    inner_circ_y = 1 .- sin.(t_inner) .- 0.5

    data = [outer_circ_x outer_circ_y ; inner_circ_x inner_circ_y]
    permutedims(data, (2, 1))
end

make_moons(n_samples::Int=1000) = make_moons(Random.GLOBAL_RNG, n_samples)

function make_s_curve(rng::AbstractRNG, n_samples::Int=1000)
    t = 3 * π * (rand(rng, n_samples) .- 0.5)
    x = sin.(t)
    y = sign.(t) .* (cos.(t) .- 1)

    permutedims([x y], (2, 1))
end

make_s_curve(n_samples::Int=1000) = make_s_curve(Random.default_rng(), n_samples)

function add_noise!(rng, x, scale=1.0)
    x += randn(rng, size(x)) * scale
end
