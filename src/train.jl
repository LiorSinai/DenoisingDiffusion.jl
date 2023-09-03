
using Flux: update!, DataLoader
using Printf: @sprintf
using ProgressMeter

function train!(loss, model, data::DataLoader, opt_state, val_data;
    num_epochs::Int=10,
    save_after_epoch::Bool=false,
    save_dir::String="",
    prob_uncond::Float64=0.0,
    )
    history = Dict(
        "epoch_size" => length(data),
        "mean_batch_loss" => Float64[],
        "val_loss" => Float64[],
        "batch_size" => data.batchsize,
    )
    for epoch = 1:num_epochs
        print(stderr, "") # clear stderr for Progress
        progress = Progress(length(data); desc="epoch $epoch/$num_epochs")
        total_loss = 0.0
        for (idx, x) in enumerate(data)
            if (x isa Tuple)
                y = randomly_set_unconditioned(x[2]; prob_uncond=prob_uncond) 
                x_splat = (x[1], y)
            else
                x_splat = (x,)
            end
            batch_loss, grads = Flux.withgradient(model) do m
                loss(m, x_splat...)
            end
            total_loss += batch_loss
            ProgressMeter.next!(progress; showvalues=[("batch loss", @sprintf("%.5f", batch_loss))])
            Flux.update!(opt_state, model, grads[1])
        end
        if save_after_epoch
            path = joinpath(save_dir, "model_epoch=$(epoch).bson")
            let model = cpu(model) # keep main model on device
                BSON.bson(path, Dict(:model => model))
            end
        end
        push!(history["mean_batch_loss"], total_loss / length(data))
        @printf("mean batch loss: %.5f ; ", history["mean_batch_loss"][end])
        update_history!(model, history, loss, val_data; prob_uncond=prob_uncond)
    end
    history
end

function randomly_set_unconditioned(
    labels::AbstractVector{Int}; prob_uncond::Float64=0.20
    )
    # with probability prob_uncond we train without class conditioning
    if prob_uncond == 0.0
        return embeddings
    end
    labels = copy(labels)
    batch_size = length(labels)
    is_not_class_cond = rand(batch_size) .<= prob_uncond
    labels[is_not_class_cond] .= 1
    labels
end

function update_history!(model, history, loss, val_data; prob_uncond::Float64=0.0)
    val_loss = batched_loss(loss, model, val_data; prob_uncond=prob_uncond)
    push!(history["val_loss"], val_loss)
    @printf("val loss: %.5f", history["val_loss"][end])
    println("")
end

function batched_loss(loss, model, data::DataLoader; prob_uncond::Float64=0.0)
    total_loss = 0.0
    for x in data
        if (x isa Tuple)
            y = randomly_set_unconditioned(x[2]; prob_uncond=prob_uncond) 
            x_splat = (x[1], y)
        else
            x_splat = (x,)
        end
        total_loss += loss(model, x_splat...)
    end
    total_loss /= length(data)
end

"""
    split_validation(rng, data[, labels]; frac=0.1)

Splits `data` and `labels` into two datasets of size `1-frac` and `frac` respectively.

Warning: this function duplicates `data`.
"""
function split_validation(rng::AbstractRNG, data::AbstractArray; frac=0.1)
    nsamples = size(data)[end]
    idxs = randperm(rng, nsamples)
    ntrain = nsamples - floor(Int, frac * nsamples)
    inds_start = ntuple(Returns(:), ndims(data) - 1)
    train_data = data[inds_start..., idxs[1:ntrain]]
    val_data = data[inds_start..., idxs[(ntrain + 1):end]]
    train_data, val_data
end

function split_validation(rng::AbstractRNG, data::AbstractArray, labels::AbstractVecOrMat; frac=0.1)
    nsamples = size(data)[end]
    idxs = randperm(rng, nsamples)
    ntrain = nsamples - floor(Int, frac * nsamples)
    inds_start = ntuple(Returns(:), ndims(data) - 1)
    ## train data
    idxs_train = idxs[1:ntrain]
    train_data = data[inds_start..., idxs_train]
    train_labels = ndims(labels) == 2 ? labels[:, idxs_train] : labels[idxs_train]
    ## validation data
    idxs_val = idxs[(ntrain + 1):end]
    val_data = data[inds_start..., idxs_val]
    val_labels = ndims(labels) == 2 ? labels[:, idxs_val] : labels[idxs_val]
    (train_data, train_labels), (val_data, val_labels)
end

"""
    batched_metric(g, f, data::DataLoader, g=identity)

Caculates `f(g(x), y)` for each `(x, y)` in data and returns a weighted sum by batch size.
If `f` takes the mean this will recover the full sample mean.
Reduces memory load for `f` and `g`. 
"""
function batched_metric(g, f, data::DataLoader)
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

count_observations(data::D) where {D<:DataLoader} = count_observations(data.data)
count_observations(data::Tuple) = count_observations(data[1])
count_observations(data::AbstractArray{<:Any,N}) where {N} = size(data, N)
count_observations(data) = length(data)