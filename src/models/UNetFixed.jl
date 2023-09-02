import Flux._big_show
using Flux: _channels_in, _channels_out, _big_finale, _layer_show

"""
    UNetFixed(in_channels, model_channels, num_timesteps; 
        block_layer=ResBlock,
        block_groups=8,
        num_attention_heads=4,
    )

A 14 layer convolutional autoencoder with time embeddings (each ResBlock and upsample has 2 layers).
Each downsample halves the image dimensions so it should only be used on even sized images.

The model grows in size with `model_channels^2`.

Key word arguments:
- `channel_multipliers`: the multiplification factor on `model_channels` on each down layer.
- `block_layer`: the main layer for down and up blocks.
- `block_groups`: a parameter for the `block_layer` for group normalization.
- `num_attention_heads`: number of attention heads for multi-head attention. Should evenly divide the model channels at the this point.
    That will be `model_channel*channel_multipliers[end]` in the middle.

```
          +----+     +-----+     +----------+     +-----+     +----------+     +-----+     +----+
downs     |Conv| --> |Block| --> |Downsample| --> |Block| --> |Downsample| --> |Block| --> |Conv|
          +----+     +-----+  |  +----------+     +-----+  |  +----------+     +-----+  |  +----+
                              |                            |                            |    |
                              |                            |                            |  +------+
middle                        |                            |                            |  |middle|
                              |                            |                            |  +------+
          +----+     +-----+  v   +--------+      +-----+  v   +--------+      +-----+  v    |
ups       |Conv| <-- |Block| <--- |Upsample| <--- |Block| <--- |Upsample| <--- |Block| <-----|
          +----+     +-----+      +--------+      +-----+      +--------+      +-----+
```
"""
struct UNetFixed{E,D<:Tuple,M<:Tuple,U<:Tuple}
    time_embedding::E
    downs::D
    middle::M
    ups::U
end

Flux.@functor UNetFixed

function UNetFixed(
    in_channels::Int,
    model_channels::Int,
    num_timesteps::Int
    ;
    block_layer=ResBlock,
    block_groups::Int=8,
    num_attention_heads::Int=4
    #num_blocks::Int=1, ##TODO
    )
    model_channels % block_groups == 0 ||
        error("The number of block_groups ($(block_groups)) must divide the number of model_channels ($model_channels)")

    channel_multipliers = (1, 2, 4) # hardcoded because Zygote does not support mutating arrays
    channels = [model_channels, map(m -> model_channels * m, channel_multipliers)...]
    in_out = collect(zip(channels[1:end-1], channels[2:end]))

    time_dim = 4 * model_channels
    time_embed = Chain(
        SinusoidalPositionEmbedding(num_timesteps, time_dim),
        Dense(time_dim, time_dim, gelu),
        Dense(time_dim, time_dim)
    )

    downs = Any[
        Conv((3, 3), in_channels => model_channels, stride=(1, 1), pad=(1, 1))
    ]
    for (level, (in_ch, out_ch)) in enumerate(in_out)
        is_last = level == length(in_out)
        block = block_layer(in_ch => in_ch, time_dim; groups=block_groups)
        push!(downs, block)
        if !is_last
            block = downsample_layer(in_ch => out_ch)
            push!(downs, block)
        else
            block = Conv((3, 3), in_ch => out_ch, stride=(1, 1), pad=(1, 1))
            push!(downs, block)
        end
    end

    mid_ch = channels[end]
    middle = (
        block_layer(mid_ch => mid_ch, time_dim; groups=block_groups),
        SkipConnection(MultiheadAttention(mid_ch, nhead=num_attention_heads), +),
        block_layer(mid_ch => mid_ch, time_dim; groups=block_groups),
    )

    ups = Any[]
    for (level, (in_ch, out_ch)) in enumerate(reverse(in_out))
        is_last = level == length(in_out)
        block = block_layer((in_ch + out_ch) => out_ch, time_dim; groups=block_groups)
        push!(ups, block)
        if !is_last
            block = upsample_layer(out_ch => in_ch)
            push!(ups, block)
        end
    end

    final_conv = Conv((3, 3), model_channels => in_channels, stride=(1, 1), pad=(1, 1))
    push!(ups, final_conv)

    UNetFixed(time_embed, tuple(downs...), middle, tuple(ups...))
end

function (u::UNetFixed)(x::AbstractArray, timesteps::AbstractVector{Int})
    if (size(x, 1) % 4 != 0) || (size(x, 2) % 4 != 0)
        throw(DimensionMismatch(
            "image sizes $(size(x)[1:2]) is not divisible by 8, which is required for concatenation during upsampling.")
        )
    end
    emb = u.time_embedding(timesteps)

    h = x
    ## downs
    h = _maybe_forward(u.downs[1], h, emb) # init
    h = _maybe_forward(u.downs[2], h, emb) # block
    h2 = h
    h = _maybe_forward(u.downs[3], h, emb) # downsample
    h = _maybe_forward(u.downs[4], h, emb) # block
    h4 = h
    h = _maybe_forward(u.downs[5], h, emb) # downsample
    h = _maybe_forward(u.downs[6], h, emb) # block
    h6 = h
    h = _maybe_forward(u.downs[7], h, emb) # block
    for layer in u.middle
        h = _maybe_forward(layer, h, emb)
    end
    ## ups
    h = cat_on_channel_dim(h, h6)
    h = _maybe_forward(u.ups[1], h, emb) # block
    h = _maybe_forward(u.ups[2], h, emb) # upsample
    h = cat_on_channel_dim(h, h4)
    h = _maybe_forward(u.ups[3], h, emb) # block
    h = _maybe_forward(u.ups[4], h, emb) # upsample
    h = cat_on_channel_dim(h, h2)
    h = _maybe_forward(u.ups[5], h, emb) # block
    h = _maybe_forward(u.ups[6], h, emb) # final
    h
end

## show

function Base.show(io::IO, u::UNetFixed)
    print(io, "UNetFixed(")
    print(io, "time_embedding=", u.time_embedding)
    for level in [:downs, :middle, :ups]
        print(io, ", ", level, "=(")
        for (i, layer) in enumerate(getproperty(u, level))
            i != 1 && print(io, " ")
            print(io, _min_str(layer), ",")
        end
        print(")")
    end
    print(io, ")")
end

_min_str(l) = string(l)
function _min_str(l::Conv)
    stride = l.stride[1]
    filter = size(l.weight, 1)
    "Conv($(_channels_in(l)) => $(_channels_out(l)), f=$filter" * (stride == 1 ? ")" : ", s=$stride)")
end
_min_str(l::ConvEmbed) = "ConvEmbed($(_channels_in(l.conv)) => $(_channels_out(l.conv)))"
_min_str(l::ResBlock) = "ResBlock($(_channels_in(l.in_layers.conv)) => $(_channels_out(l.in_layers.conv)))"
_min_str(l::Chain) = "Chain(" * join([_min_str(x) for x in l], ", ") * ")"

function _big_show(io::IO, u::UNetFixed, indent::Int=0, name=nothing)
    println(io, " "^indent, isnothing(name) ? "" : "$name = ", "UNetFixed(")
    for layer in [:time_embedding, :downs, :middle, :ups]
        _big_show(io, getproperty(u, layer), indent + 2, layer)
    end
    if indent == 0
        print(io, ") ")
        _big_finale(io, u)
    else
        println(io, " "^indent, ")", ",")
    end
end

function Base.show(io::IO, m::MIME"text/plain", x::UNetFixed)
    if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
        _big_show(io, x)
    elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
        _layer_show(io, x)
    else
        show(io, x)
    end
end
