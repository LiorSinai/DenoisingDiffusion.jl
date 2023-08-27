"""
    SinusoidalPositionEmbedding(dim_embedding::Int, max_length::Int=1000)

A position encoding layer for a matrix of size `dim_embedding`. `max_len` is the maximum acceptable length of input.

For each a pair of rows `(2i, 2i+1)` and a position `k`, the encoding is calculated as:

    W[2i, k] = sin(pos/(1e4^(2i/dim_embedding)))
    W[2i + 1, k] = cos(pos/(1e4^(2i/dim_embedding)))

"""
struct SinusoidalPositionEmbedding{W<:AbstractArray}
    weight::W
end

Flux.@functor SinusoidalPositionEmbedding
Flux.trainable(emb::SinusoidalPositionEmbedding) = (;) # not trainable

function SinusoidalPositionEmbedding(in::Int, out::Int)
    W = make_positional_embedding(out, in)
    SinusoidalPositionEmbedding(W)
end

function make_positional_embedding(dim_embedding::Int, seq_length::Int=1000; n::Int=10000)
    embedding = Matrix{Float32}(undef, dim_embedding, seq_length)
    for pos in 1:seq_length
        for row in 0:2:(dim_embedding-1)
            denom = 1.0 / (n^(row / (dim_embedding - 2)))
            embedding[row+1, pos] = sin(pos * denom)
            embedding[row+2, pos] = cos(pos * denom)
        end
    end
    embedding
end

(m::SinusoidalPositionEmbedding)(x::Integer) = m.weight[:, x]
(m::SinusoidalPositionEmbedding)(x::AbstractVector) = NNlib.gather(m.weight, x)
(m::SinusoidalPositionEmbedding)(x::AbstractArray) = reshape(m(vec(x)), :, size(x)...)

function Base.show(io::IO, m::SinusoidalPositionEmbedding)
    print(io, "SinusoidalPositionEmbedding(", size(m.weight, 2), " => ", size(m.weight, 1), ")")
end
