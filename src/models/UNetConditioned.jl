import Flux._big_show
using Flux: _big_finale, _layer_show

"""
    UNetConditioned(in_channels, model_channels, num_timesteps; 
        channel_multipliers=(1, 2, 4),
        block_layer=ResBlock,
        block_groups=8, 
        middle_attention=true,
        num_attention_heads=4,
        combine_embeddings=vcat,
        num_classes=1, 
    )

A convolutional autoencoder with time embeddings, class embeddings and skip connections.
The default configuration has 17 layers and skip connections (each ResBlock and upsample has 2 layers).
Each downsample halves the image dimensions so it should only be used on even sized images.

The model grows in size with `model_channels^2`.

Key word arguments:
- `channel_multipliers`: the multiplification factor on `model_channels` on each down layer.
- `block_layer`: the main layer for down and up blocks.
- `block_groups`: a parameter for the `block_layer` for group normalization.
- `middle_attention`: have one attention layer in the middle of the model. This is a large and expensive layer.
- `num_attention_heads`: number of attention heads for multi-head attention. Should evenly divide the model channels at the this point.
    That will be `model_channel*channel_multipliers[end]` in the middle.
- `combine_embeddings`: how to combine time and class embeddings. Recommendations are `vcat` or `.+` or `.*`.
- `num_classes`: number of class embeddings.

```
+-----+     +-------+     +-------+     +-----+     +------+
|:init| --> |:down_1| --> |:skip_1| --> |:up_1| --> |:final|
+-----+     +-------+     +-------+     +-----+     +------+
                            |   |
        ---------------------   -----------------------------------
        |                                                         |
 +-------------+     +-------+     +-------+     +-----+     +-----------+
 |:downsample_1| --> |:down_2| --> |:skip_2| --> |:up_2| --> |:upsample_2|
 +-------------+     +-------+     +-------+     +-----+     +-----------+
                                     |   |
        ------------------------------   --------------------------
        |                                                         |
 +-------------+     +-------+     +-------+     +-----+     +-----------+
 |:downsample_2| --> |:down_3| --> |:skip_3| --> |:up_3| --> |:upsample_3|
 +-------------+     +-------+     +-------+     +-----+     +-----------+
                                     |   |
                                ------   ------
                                |             |
                            +-------+     +-------+
                            |:down_4| --> |:middle|
                            +-------+     +-------+
```
"""
struct UNetConditioned{E1,E2,F,C<:ConditionalChain}
    time_embedding::E1
    class_embedding::E2
    combine_embeddings::F
    chain::C
    num_levels::Int
end

Flux.@functor UNetConditioned (time_embedding, class_embedding, chain,)

function UNetConditioned(
    in_channels::Int,
    model_channels::Int,
    num_timesteps::Int,
    ;
    num_classes::Int=1,
    channel_multipliers::NTuple{N,Int}=(1, 2, 4),
    block_layer=ResBlock,
    num_blocks_per_level::Int=1,
    block_groups::Int=8,
    num_attention_heads::Int=4,
    middle_attention::Bool=true,
    combine_embeddings=vcat
    ) where {N}
    model_channels % block_groups == 0 ||
        error("The number of block_groups ($(block_groups)) must divide the number of model_channels ($model_channels)")

    channels = [model_channels, map(m -> model_channels * m, channel_multipliers)...]
    in_out = collect(zip(channels[1:end-1], channels[2:end]))

    time_dim = 4 * model_channels
    time_embed = Chain(
        SinusoidalPositionEmbedding(num_timesteps, time_dim),
        Dense(time_dim, time_dim, gelu),
        Dense(time_dim, time_dim)
    )
    class_embedding = Flux.Embedding((num_classes + 1) => time_dim)
    embed_dim = (combine_embeddings == vcat) ? 2 * time_dim : time_dim

    in_ch, out_ch = in_out[1]
    down_keys = num_blocks_per_level == 1 ? [Symbol("down_1")] : [Symbol("down_1_$(i)") for i in 1:num_blocks_per_level]
    up_keys = num_blocks_per_level == 1 ? [Symbol("up_1")] : [Symbol("up_1_$(i)") for i in 1:num_blocks_per_level]
    down_blocks = [
        block_layer(in_ch => in_ch, embed_dim; groups=block_groups) for i in 1:num_blocks_per_level
    ]
    up_blocks = [
        block_layer((in_ch + out_ch) => out_ch, embed_dim; groups=block_groups),
        [block_layer(out_ch => out_ch, embed_dim; groups=block_groups) for i in 2:num_blocks_per_level]...
    ]
    chain = ConditionalChain(;
        init=Conv((3, 3), in_channels => model_channels, stride=(1, 1), pad=(1, 1)),
        NamedTuple(zip(down_keys, down_blocks))...,
        skip_1=ConditionalSkipConnection(
            _add_unet_level(in_out, embed_dim, 2;
                block_layer=block_layer,
                block_groups=block_groups,
                num_attention_heads=num_attention_heads,
                num_blocks_per_level=num_blocks_per_level,
                middle_attention=middle_attention,
            ),
            cat_on_channel_dim
        ),
        NamedTuple(zip(up_keys, up_blocks))...,
        final=Conv((3, 3), model_channels => in_channels, stride=(1, 1), pad=(1, 1))
    )

    UNetConditioned(time_embed, class_embedding, combine_embeddings, chain, length(channel_multipliers) + 1)
end

function (u::UNetConditioned)(x::AbstractArray, timesteps::AbstractVector{Int}, labels::AbstractVector{Int})
    downsize_factor = 2^(u.num_levels - 2)
    if (size(x, 1) % downsize_factor != 0) || (size(x, 2) % downsize_factor != 0)
        throw(DimensionMismatch(
            "image size $(size(x)[1:2]) is not divisible by $downsize_factor which is required for concatenation during upsampling.")
        )
    end
    time_emb = u.time_embedding(timesteps)
    class_emb = u.class_embedding(labels)
    emb = u.combine_embeddings(time_emb, class_emb)
    h = u.chain(x, emb)
    h
end

function (u::UNetConditioned)(x::AbstractArray, timesteps::AbstractVector{Int})
    batch_size = length(timesteps)
    labels = fill(1, batch_size)
    u(x, timesteps, labels)
end

## show

function Base.show(io::IO, u::UNetConditioned)
    print(io, "UNetConditioned(")
    print(io, "time_embedding=", u.time_embedding)
    print(io, "class_embedding=", u.class_embedding)
    print(io, "combine_embeddings=", u.combine_embeddings)
    print(io, ", chain=", u.chain)
    print(io, ")")
end

function _big_show(io::IO, u::UNetConditioned, indent::Int=0, name=nothing)
    println(io, " "^indent, isnothing(name) ? "" : "$name = ", "UNetConditioned(")
    for layer in [:time_embedding, :class_embedding, :combine_embeddings, :chain]
        _big_show(io, getproperty(u, layer), indent + 2, layer)
    end
    if indent == 0
        print(io, ") ")
        _big_finale(io, u)
    else
        println(io, " "^indent, ")", ",")
    end
end

function Base.show(io::IO, m::MIME"text/plain", x::UNetConditioned)
    if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
        _big_show(io, x)
    elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
        _layer_show(io, x)
    else
        show(io, x)
    end
end
