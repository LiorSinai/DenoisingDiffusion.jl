
using Flux: update!, DataLoader
using Flux.Optimise: AbstractOptimiser
using Flux.Zygote: sensitivity, pullback
using Printf: @sprintf

function train!(loss, diffusion::GaussianDiffusion, data, opt::AbstractOptimiser, val_data;
    num_epochs::Int=10,
    save_after_epoch::Bool=false,
    save_dir::String=""
)
    history = Dict(
        "epoch_size" => count_observations(data),
        "train_loss" => Float64[],
        "val_loss" => Float64[],
    )
    for epoch = 1:num_epochs
        losses = Vector{Float64}()
        progress = Progress(length(data); desc="epoch $epoch/$num_epochs")
        params = Flux.params(diffusion) # keep here in case of data movement between devices (this might happen during saving)
        for x in data
            batch_loss, back = pullback(params) do
                loss(diffusion, x)
            end
            grads = back(sensitivity(batch_loss))
            Flux.update!(opt, params, grads)
            push!(losses, batch_loss)
            ProgressMeter.next!(progress; showvalues=[("batch loss", @sprintf("%.5f", batch_loss))])
        end
        if save_after_epoch
            path = joinpath(save_dir, "diffusion_epoch=$(epoch).bson")
            let diffusion = cpu(diffusion) # keep main diffusion on device
                BSON.bson(path, Dict(:diffusion => diffusion))
            end
        end
        update_history!(diffusion, history, loss, losses, val_data)
    end
    history
end

count_observations(data::D) where {D<:DataLoader} = count_observations(data.data)
count_observations(data::Tuple) = count_observations(data[1])
count_observations(data::AbstractArray{<:Any,N}) where {N} = size(data, N)
count_observations(data) = length(data)

function update_history!(diffusion, history, loss, train_losses, val_data)
    push!(history["train_loss"], sum(train_losses) / length(train_losses))

    val_loss = 0.0
    for x in val_data
        val_loss += loss(diffusion, x)
    end
    push!(history["val_loss"], val_loss / length(val_data))

    @printf("train loss: %.5f ; ", history["train_loss"][end])
    @printf("val loss: %.5f", history["val_loss"][end])
    println("")
end

function split_validation(rng::AbstractRNG, data::AbstractArray; frac=0.1)
    nsamples = size(data)[end]
    idxs = randperm(rng, nsamples)
    ntrain = nsamples - floor(Int, frac * nsamples)
    data[:, :, :, idxs[1:ntrain]], data[:, :, :, idxs[(ntrain+1):end]]
end

function split_validation(rng::AbstractRNG, data::AbstractArray, labels::AbstractVector{Int}; frac=0.1)
    nsamples = size(data)[end]
    idxs = randperm(rng, nsamples)
    ntrain = nsamples - floor(Int, frac * nsamples)
    train_data = (data[:, :, :, idxs[1:ntrain]], labels[idxs[1:ntrain]])
    val_data = (data[:, :, :, idxs[(ntrain+1):end]], labels[idxs[(ntrain+1):end]])
    train_data, val_data
end

"""
    batched_metric(f, data, g=identity)

Caculates `f(g(x), y)` for each `(x, y)` in data and returns a weighted sum by batch size.
If `f` takes the mean this will recover the full sample mean.
Reduces memory load for `f` and `g`. 
To automatically batch data, use `Flux.DataLoader`.
"""
function batched_metric(f, data::DataLoader, g=identity)
    result = 0.0
    num_observations = 0
    for (x, y) in data
        metric = f(g(x), y) 
        batch_size = count_observations(x) 
        result += metric * batch_size
        num_observations += batch_size
    end
    result / num_observations
end

"""
    load_opt_state!(opt, params_src, params_dest; to_device=cpu)

The optimiser state and the model are tightly coupled. 
This is problematic for the case where data is moved between a CPU and a GPU.
A careful sequence needs to be followed with serialising and deserialising so that they stay in sync.

Sequence to save:
1) `params_gpu = Flux.params(model)`;
2) `model_cpu = cpu(model)`
3) `load_opt_state!(opt, params_gpu, Flux.params(model_cpu), to_device=cpu)`
4) `BSON.bson(path, Dict(:model => model_cpu, :opt => opt))` 

Sequence to load:
1) `BSON.@load path model opt`
2) `params_cpu = Flux.params(model);`
3) `model = gpu(model)`
4) `load_opt_state!(opt, params_cpu, Flux.params(model), to_device=gpu)`

See https://discourse.julialang.org/t/deepcopy-flux-model/72930
"""
function load_opt_state!(opt::Adam, params_src, params_dest; to_device=cpu)
    state = IdDict()
    for (p_dest, p_src) in zip(params_dest, params_src)
        mt, vt, βp = opt.state[p_src]
        state[p_dest] = (to_device(mt), to_device(vt), βp,)
    end
    opt.state = state
end
