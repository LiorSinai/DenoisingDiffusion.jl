using DenoisingDiffusion
using Test
using Flux

@testset verbose = true "DenoisingDiffusion" begin
    include("test_models.jl")
    include("convert_fixed_nested.jl")
    include("test_attention.jl")
    include("ddim.jl")
    include("split_validation.jl")
end