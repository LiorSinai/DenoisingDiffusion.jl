using Flux: _channels_in, _channels_out, _big_finale, _layer_show
import Flux._big_show

struct MultiheadAttention{Q<:Conv,O<:Conv}
    nhead::Int
    to_qkv::Q
    to_out::O
end

Flux.@functor MultiheadAttention (to_qkv, to_out,)

"""
    MultiheadAttention(nhead::Int, dim_model::Int)
    MultiheadAttention(nhead::Int, dim_model::Int, dim_head::Int)

Multihead dot product attention Layer.

The attention heads are split across channels `C` whereas in a text transformers they are split across the embedding layer. 
The input here is `W × H × C × B` => `WH × C × B` compared to the usual `dm × N × B` input of a text transformer. 
Here the second dimension is split: `C=nhead*dim_head` whereas in a text transformer, the first dimension is usually split `dm=nhead*dim_head`.
"""
function MultiheadAttention(dim_model::Int, dim_head::Int; nhead::Int=4)
    dim_hidden = dim_head * nhead
    MultiheadAttention(
        nhead,
        Conv((3, 3), dim_model => dim_hidden * 3, stride=(1, 1), pad=(1, 1), bias=false),
        Conv((3, 3), dim_hidden => dim_model, stride=(1, 1), pad=(1, 1))
    )
end

function MultiheadAttention(dim_model::Int; nhead::Int=4)
    if dim_model % nhead != 0
        error("model dimension=$dim_model is not divisible by number of heads=$nhead")
    end
    MultiheadAttention(dim_model, div(dim_model, nhead), nhead=nhead)
end

function (mha::MultiheadAttention)(x::A) where {T,A<:AbstractArray{T,4}}
    # batch multiplication version. Input is W × H × C × B
    qkv = mha.to_qkv(x)
    Q, K, V = Flux.chunk(qkv, 3, dims=3)

    c = size(Q, 3)
    dh = div(c, mha.nhead)
    #size(Q) == (W, H, dh*nhead, B) => (W*H, dh, nhead, B) => (dh, W*H, nhead, B)
    Q = permutedims(reshape(Q, :, dh, mha.nhead, size(x, 4)), [2, 1, 3, 4])
    K = permutedims(reshape(K, :, dh, mha.nhead, size(x, 4)), [2, 1, 3, 4])
    V = permutedims(reshape(V, :, dh, mha.nhead, size(x, 4)), [2, 1, 3, 4])
    #size(attn) == (dh, W*H, nhead, B)
    attn = scaled_dot_attention(Q, K, V)
    #size(attn) == (dh, W*H, nhead, B) => (W*H, dh, nhead, B) => (W, H, dh*nhead, B)
    attn = permutedims(attn, [2, 1, 3, 4])
    attn = reshape(attn, size(x, 1), size(x, 2), c, size(x, 4))

    mha.to_out(attn)
end

function (mha::MultiheadAttention)(x::A) where {T,A<:AbstractArray{T,3}}
    # single sample. Make it a batch of 1
    x = reshape(x, size(x)..., 1)
    attn = mha(x)
    reshape(attn, size(attn)[1:end-1]...)
end

"""
    scaled_dot_attention(query, key, value)

Scaled dot attention as proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 

If the inputs are matrices, the output is:
    
    A = 1/sqrt(dh) * value * softmax(transpose(key) * query))

If the inputs are 3D arrays, the output is

    A[:, :, h] = 1/sqrt(dh) * value[:, :, h] * softmax(transpose(key[:, :, h]) * query[:, :, h]))   

If the inputs are 4D arrays, the output is 
    
    A[:, :, h, b] = 1/sqrt(dh) * value[:, :, h, b] * softmax(transpose(key[:, :, h, b]) * query[:, :, h, b]))
"""
function scaled_dot_attention(query::A1, key::A2, value::A3) where {
    T,A1<:AbstractArray{T,4},A2<:AbstractArray{T,4},A3<:AbstractArray{T,4}}
    # Batched version. Input is (dh, N, nhead, B)
    dh = size(query, 1)
    scale = one(T) / convert(T, sqrt(dh))
    keyT = permutedims(key, (2, 1, 3, 4)) # important: don't use a view (PermutedDimsArray) because this slows batched_mul
    sim = scale .* batched_mul(keyT, query) #size(sim) == (N, N, nhead, B)
    sim = softmax(sim; dims=1)
    batched_mul(value, sim) #size(attention) == (dh, N, nhead, B)
end

function scaled_dot_attention(query::A1, key::A2, value::A3) where {
    T,A1<:AbstractArray{T,3},A2<:AbstractArray{T,3},A3<:AbstractArray{T,3}}
    # Input is (dh, N, nhead)
    dh = size(query, 1)
    scale = one(T) / convert(T, sqrt(dh))
    keyT = permutedims(key, (2, 1, 3))
    sim = scale .* batched_mul(keyT, query) #size(sim) == (N, N, nhead)
    sim = softmax(sim; dims=1)
    batched_mul(value, sim)  #size(attention) == (dh, N, nhead) 
end

function scaled_dot_attention(query::A1, key::A2, value::A3) where {
    T,A1<:AbstractMatrix{T},A2<:AbstractMatrix{T},A3<:AbstractMatrix{T}}
    ## Matrix version for a single head. Input is (dh, N)
    dh = size(query, 1)
    scale = one(T) / convert(T, sqrt(dh))
    sim = scale .* transpose(key) * query #size(sim) == (N, N)
    sim = softmax(sim; dims=1)
    value * sim #size(attention) == (dh, N)
end

## show

function Base.show(io::IO, mha::MultiheadAttention)
    dim_head = _channels_out(mha.to_qkv) ÷ 3 ÷ mha.nhead
    dim_model = _channels_in(mha.to_qkv)
    print(io, "MultiheadAttention(")
    print(io, "nhead=$(mha.nhead), ")
    print(io, "head_size=$(dim_head), ")
    print(io, "$(dim_model)=>$(dim_model)")
    print(io, ")")
end

function _big_show(io::IO, m::MultiheadAttention, indent::Int=0, name=nothing)
    println(io, " "^indent, isnothing(name) ? "" : "$name = ", "MultiheadAttention(")
    for layer in [:nhead, :to_qkv, :to_out]
        _big_show(io, getproperty(m, layer), indent + 2, layer)
    end
    if indent == 0
        print(io, ") ")
        _big_finale(io, m)
    else
        println(io, " "^indent, ")", ",")
    end
end

function Base.show(io::IO, m::MIME"text/plain", x::MultiheadAttention)
    if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
        _big_show(io, x)
    elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
        _layer_show(io, x)
    else
        show(io, x)
    end
end