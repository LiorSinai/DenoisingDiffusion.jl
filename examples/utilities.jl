using StatsBase
using LinearAlgebra
using Images
using Optimisers: Leaf

## model sizes 

function conv_output_size(input_size::Int, filter_size::Int, stride::Int=1, pad::Int=0)
    floor(Int, (input_size + 2 * pad - filter_size) / stride) + 1
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

unnormalize_zero_to_one(x::AbstractArray) = (x .+ one(x[1])) / 2;

"""
    img_WHC_to_rgb(img_WHC) where {T}

Converts images in (width, height, channel) form to RGB images.
Should normalize the images first for better constrasts.
"""
function img_WHC_to_rgb(img_WHC::AbstractArray{T,N}) where {T,N}
    @assert N == 3 || N == 4
    @assert size(img_WHC, 3) == 3
    img_CHW = permutedims(img_WHC, (3, 2, 1, 4:N...))
    img = Images.colorview(Images.RGB, img_CHW)
    img
end

"""
    img_WH_to_gray(img_WH) where {T}

Converts images in (width, height) form to gray images.
"""
function img_WH_to_gray(img_WH::AbstractArray{T,N}) where {T,N}
    @assert N == 2 || N == 3
    img_HW = permutedims(img_WH, (2, 1, 3:N...))
    img = Images.colorview(Images.Gray, img_HW)
    img
end

## optimisers

function extract_rule_from_tree(tree::Union{NamedTuple, Tuple})
    for state in tree
        rule = extract_rule_from_tree(state)
        if !isnothing(rule)
            return rule
        end
    end
end

extract_rule_from_tree(leaf::Leaf) = leaf.rule
