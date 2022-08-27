using Plots
using Flux
using Flux: Embedding
using Dates
using BSON, JSON
using Printf

using DenoisingDiffusion
using DenoisingDiffusion: train!
include("datasets.jl")
include("utilities.jl")

## settings

directory = "outputs\\" * Dates.format(now(), "yyyymmdd_HHMM")
num_timesteps = 40
n_batch = 10_000
to_device = cpu
d_hid = 16
num_epochs = 2

## data

X = normalize_neg_one_to_one(make_spiral(n_batch))

## model

model = ConditionalChain(
    Biparallel(.+, Dense(2, d_hid), Chain(SinusoidalPositionEmbedding(num_timesteps, d_hid), Dense(d_hid, d_hid))),
    swish,
    Biparallel(.+, Dense(d_hid, d_hid), Chain(SinusoidalPositionEmbedding(num_timesteps, d_hid), Dense(d_hid, d_hid))),
    swish,
    Biparallel(.+, Dense(d_hid, d_hid), Chain(SinusoidalPositionEmbedding(num_timesteps, d_hid), Dense(d_hid, d_hid))),
    swish,
    Dense(d_hid, 2),
)
display(model)

βs = linear_beta_schedule(num_timesteps, 8e-6, 9e-5)
diffusion = GaussianDiffusion(Vector{Float32}, βs, (2,), model)

### train
diffusion = diffusion |> to_device

data = Flux.DataLoader(X |> to_device; batchsize=32, shuffle=true);
X_val = normalize_neg_one_to_one(make_spiral(floor(Int, 0.1 * n_batch)))
val_data = Flux.DataLoader(X_val |> to_device; batchsize=32, shuffle=false);
loss_type = Flux.mse;
loss(diffusion, x) = p_lossess(diffusion, loss_type, x; to_device=to_device)
opt = Adam(0.001);

start_time = time_ns()
history = train!(loss, diffusion, data, opt, val_data; num_epochs=num_epochs)
end_time = time_ns() - start_time
println("\ndone training")
@printf "time taken: %.2fs\n" end_time / 1e9

### save results

mkdir(directory)
output_path = joinpath(directory, "diffusion.bson")
history_path = joinpath(directory, "history.json")
hyperparameters_path = joinpath(directory, "hyperparameters.json")

diffusion = diffusion |> cpu
BSON.bson(output_path, Dict(:diffusion=>diffusion))

open(history_path, "w") do f
    JSON.print(f, history)
end

hyperparameters = Dict(
    "num_timesteps" => num_timesteps,
    "data_shape" => "$(diffusion.data_shape)",
    "denoise_fn" => "$(typeof(model).name.wrapper)",
    "parameters" => sum(length, Flux.params(model)),
    "loss_type" => "$loss_type",
    "d_hid" => d_hid,
    "seed" => seed,
)

open(hyperparameters_path, "w") do f
    JSON.print(f, hyperparameters)
end

### plot results
diffusion = diffusion |> cpu

p = plot(1:length(history["train_loss"]), history["train_loss"], label="train_loss")
plot!(p, 1:length(history["val_loss"]), history["val_loss"], label="val_loss")
display(p)
X0 = p_sample_loop(diffusion, 1000)
p = scatter(X0[1, :], X0[2, :], alpha=0.5, label="",
    aspectratio=:equal,
    xlims=(-2, 2), ylims=(-2, 2),
)
display(p)

println("press enter to finish")
readline()