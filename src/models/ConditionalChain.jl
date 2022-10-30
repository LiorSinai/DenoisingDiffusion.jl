using Flux
import Flux._show_children
import Flux._big_show


abstract type AbstractParallel end

_maybe_forward(layer::AbstractParallel, x::AbstractArray, ys::AbstractArray...) = layer(x, ys...)
_maybe_forward(layer::Parallel, x::AbstractArray, ys::AbstractArray...) = layer(x, ys...)
_maybe_forward(layer, x::AbstractArray, ys::AbstractArray...) = layer(x)

"""
    ConditionalChain(layers...)

Based off `Flux.Chain` except takes in two inputs. 
If a layer is of type `Biparallel` it uses both inputs else it uses only one.
The first input can therefore be conditioned on the second input.
"""
struct ConditionalChain{T<:Union{Tuple, NamedTuple}} <:AbstractParallel
    layers::T
end
Flux.@functor ConditionalChain 

ConditionalChain(xs...) = ConditionalChain(xs)
function ConditionalChain(; kw...)
  :layers in keys(kw) && throw(ArgumentError("a Chain cannot have a named layer called `layers`"))
  isempty(kw) && return ConditionalChain(())
  ConditionalChain(values(kw))
end

Flux.@forward ConditionalChain.layers Base.getindex, Base.length, Base.first, Base.last,
Base.iterate, Base.lastindex, Base.keys, Base.firstindex

Base.getindex(c::ConditionalChain, i::AbstractArray) = ConditionalChain(c.layers[i]...)

function (c::ConditionalChain)(x, ys...) 
    for layer in c.layers
        x = _maybe_forward(layer, x, ys...)
    end
    x
end

function Base.show(io::IO, c::ConditionalChain)
    print(io, "ConditionalChain(")
    Flux._show_layers(io, c.layers)
    print(io, ")")
end

function _big_show(io::IO, m::ConditionalChain{T}, indent::Int=0, name=nothing) where T <: NamedTuple
    println(io, " "^indent, isnothing(name) ? "" : "$name = ", "ConditionalChain(")
    for k in Base.keys(m.layers)
        _big_show(io, m.layers[k], indent+2, k)
    end
    if indent == 0  
        print(io, ") ")
        _big_finale(io, m)
      else
        println(io, " "^indent, ")", ",")
    end
end

"""
    ConditionalSkipConnection(layers, connection)

The output is equivalent to `connection(layers(x, y), x)`.
Based off Flux.SkipConnection except passes two arguments to layers.
"""
struct ConditionalSkipConnection{T,F} <:AbstractParallel
    layers::T
    connection::F 
end
  
Flux.@functor ConditionalSkipConnection
  
function (skip::ConditionalSkipConnection)(x, ys...)
    skip.connection(skip.layers(x, ys...), x)
end

function Base.show(io::IO, b::ConditionalSkipConnection)
    print(io, "ConditionalSkipConnection(", b.layers, ", ", b.connection, ")")
end

"""
    Biparallel(connection, layer1, layer2)

The output is equivalent to `(x, y) -> connection(layer1(x), layer2(y))`.
For a more generic version with multiple arguments see `Flux.Parallel`.
"""
struct Biparallel{F,T1,T2} <:AbstractParallel
    connection::F
    layer1::T1
    layer2::T2
end

Flux.@functor Biparallel

function (m::Biparallel)(x, y)
    m.connection(m.layer1(x), m.layer2(y))
end

### Show. Copied from Flux.jl/src/layers/show.jl

for T in [
    :ConditionalChain, :Biparallel, ConditionalSkipConnection
]
    @eval function Base.show(io::IO, m::MIME"text/plain", x::$T)
        if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
            Flux._big_show(io, x)
        elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
            Flux._layer_show(io, x)
        else
            show(io, x)
        end
    end
end

_show_children(c::ConditionalChain) = c.layers
