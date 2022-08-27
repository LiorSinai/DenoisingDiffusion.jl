using DenoisingDiffusion
using Test
using DenoisingDiffusion.Flux

@testset  verbose = true  "DenoisingDiffusion" begin
    include("test_models.jl")
    include("convert_fixed_nested.jl")
end