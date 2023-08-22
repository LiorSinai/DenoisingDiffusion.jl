using DenoisingDiffusion
using Test
using Flux

@testset verbose = true "DenoisingDiffusion" begin
    include("models.jl")
    include("convert_fixed_nested.jl")
    include("attention.jl")
    include("ddim.jl")
    include("split_validation.jl")
end