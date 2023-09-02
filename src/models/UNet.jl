import Flux._big_show
using Flux: _big_finale, _layer_show

"""
    UNet(in_channels, model_channels, num_timesteps; 
        channel_multipliers=(1, 2, 4),
        block_layer=ResBlock,
        num_blocks_per_level=1,
        block_groups=8,
        middle_attention=true,
        num_attention_heads=4,
    )

A convolutional autoencoder with time embeddings and skip connections.
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
struct UNet{E,C<:ConditionalChain}
    time_embedding::E
    chain::C
    num_levels::Int
end

Flux.@functor UNet (time_embedding, chain,)

function UNet(
    in_channels::Int,
    model_channels::Int,
    num_timesteps::Int
    ;
    channel_multipliers::NTuple{N,Int}=(1, 2, 4),
    block_layer=ResBlock,
    num_blocks_per_level::Int=1,
    block_groups::Int=8,
    num_attention_heads::Int=4,
    middle_attention::Bool=true,
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

    in_ch, out_ch = in_out[1]
    down_keys = num_blocks_per_level == 1 ? [Symbol("down_1")] : [Symbol("down_1_$(i)") for i in 1:num_blocks_per_level]
    up_keys = num_blocks_per_level == 1 ? [Symbol("up_1")] : [Symbol("up_1_$(i)") for i in 1:num_blocks_per_level]
    down_blocks = [
        block_layer(in_ch => in_ch, time_dim; groups=block_groups) for i in 1:num_blocks_per_level
    ]
    up_blocks = [
        block_layer((in_ch + out_ch) => out_ch, time_dim; groups=block_groups),
        [block_layer(out_ch => out_ch, time_dim; groups=block_groups) for i in 2:num_blocks_per_level]...
    ]
    chain = ConditionalChain(;
        init=Conv((3, 3), in_channels => model_channels, stride=(1, 1), pad=(1, 1)),
        NamedTuple(zip(down_keys, down_blocks))...,
        down_1=block_layer(in_ch => in_ch, time_dim; groups=block_groups),
        skip_1=ConditionalSkipConnection(
            _add_unet_level(in_out, time_dim, 2;
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
    UNet(time_embed, chain, length(channel_multipliers) + 1)
end

function _add_unet_level(
    in_out::Vector{Tuple{Int,Int}}, 
    emb_dim::Int, 
    level::Int;
    block_layer, 
    num_blocks_per_level::Int, 
    block_groups::Int, 
    num_attention_heads::Int,
    middle_attention::Bool=true,
    )
    if level > length(in_out)
        in_ch, out_ch = in_out[end]
        keys_ = middle_attention ? 
            (Symbol("down_$level"), :middle_1, :middle_attention, :middle_2) : (Symbol("down_$level"), :middle)
        attention_layers = middle_attention ? 
            [
                SkipConnection(MultiheadAttention(out_ch, nhead=num_attention_heads), +),
                block_layer(out_ch => out_ch, emb_dim; groups=block_groups),
            ] :  []
        layers = (
            Conv((3, 3), in_ch => out_ch, stride=(1, 1), pad=(1, 1)),
            block_layer(out_ch => out_ch, emb_dim; groups=block_groups),
            attention_layers...
        )
    else # recurse down a layer
        in_ch_prev, out_ch_prev = in_out[level-1]
        in_ch, out_ch = in_out[level]
        down_keys = num_blocks_per_level == 1 ? [Symbol("down_$(level)")] : [Symbol("down_$(level)_$(i)") for i in 1:num_blocks_per_level]
        up_keys = num_blocks_per_level == 1 ? [Symbol("up_$(level)")] : [Symbol("up_$(level)_$(i)") for i in 1:num_blocks_per_level]
        keys_ = (
            Symbol("downsample_$(level-1)"),
            down_keys...,
            Symbol("skip_$level"),
            up_keys...,
            Symbol("upsample_$level")
        )
        down_blocks = [
            block_layer(in_ch => in_ch, emb_dim; groups=block_groups) for i in 1:num_blocks_per_level
        ]
        up_blocks = [
            block_layer((in_ch + out_ch) => out_ch, emb_dim; groups=block_groups),
            [block_layer(out_ch => out_ch, emb_dim; groups=block_groups) for i in 2:num_blocks_per_level]...
        ]
        layers = (
            downsample_layer(in_ch_prev => out_ch_prev),
            down_blocks...,
            ConditionalSkipConnection(
                _add_unet_level(in_out, emb_dim, level + 1;
                    block_layer=block_layer,
                    block_groups=block_groups,
                    num_attention_heads=num_attention_heads,
                    num_blocks_per_level=num_blocks_per_level,
                    middle_attention=middle_attention,
                ),
                cat_on_channel_dim
            ),
            up_blocks...,
            upsample_layer(out_ch => in_ch),
        )
    end
    ConditionalChain((; zip(keys_, layers)...))
end

function (u::UNet)(x::AbstractArray, timesteps::AbstractVector{Int})
    downsize_factor = 2^(u.num_levels - 2)
    if (size(x, 1) % downsize_factor != 0) || (size(x, 2) % downsize_factor != 0)
        throw(DimensionMismatch(
            "image size $(size(x)[1:2]) is not divisible by $downsize_factor which is required for concatenation during upsampling.")
        )
    end
    emb = u.time_embedding(timesteps)
    h = u.chain(x, emb)
    h
end

## show

function Base.show(io::IO, u::UNet)
    print(io, "UNet(")
    print(io, "time_embedding=", u.time_embedding)
    print(io, ", chain=", u.chain)
    print(io, ")")
end

function _big_show(io::IO, u::UNet, indent::Int=0, name=nothing)
    println(io, " "^indent, isnothing(name) ? "" : "$name = ", "UNet(")
    for layer in [:time_embedding, :chain]
        _big_show(io, getproperty(u, layer), indent + 2, layer)
    end
    if indent == 0
        print(io, ") ")
        _big_finale(io, u)
    else
        println(io, " "^indent, ")", ",")
    end
end

function Base.show(io::IO, m::MIME"text/plain", x::UNet)
    if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
        _big_show(io, x)
    elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
        _layer_show(io, x)
    else
        show(io, x)
    end
end
