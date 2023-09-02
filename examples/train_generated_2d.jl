using Plots
using Flux
using Dates
using BSON, JSON
using Printf

using DenoisingDiffusion
using DenoisingDiffusion: train!
include("datasets.jl")
include("utilities.jl")

### settings
directory = joinpath("outputs", "2d_" * Dates.format(now(), "yyyymmdd_HHMM"))
num_timesteps = 40
n_batch = 10_000
to_device = cpu
d_hid = 16
num_epochs = 100

### data
X = normalize_neg_one_to_one(make_spiral(n_batch))
X_val = normalize_neg_one_to_one(make_spiral(floor(Int, 0.1 * n_batch)))

### model
model = ConditionalChain(
    Parallel(.+, Dense(2, d_hid), Chain(SinusoidalPositionEmbedding(num_timesteps, d_hid), Dense(d_hid, d_hid))),
    swish,
    Parallel(.+, Dense(d_hid, d_hid), Chain(SinusoidalPositionEmbedding(num_timesteps, d_hid), Dense(d_hid, d_hid))),
    swish,
    Parallel(.+, Dense(d_hid, d_hid), Chain(SinusoidalPositionEmbedding(num_timesteps, d_hid), Dense(d_hid, d_hid))),
    swish,
    Dense(d_hid, 2),
)
display(model)

βs = linear_beta_schedule(num_timesteps, 8e-6, 9e-5)
diffusion = GaussianDiffusion(Vector{Float32}, βs, (2,), model)

### train
diffusion = diffusion |> to_device

train_data = Flux.DataLoader(X |> to_device; batchsize=32, shuffle=true);
val_data = Flux.DataLoader(X_val |> to_device; batchsize=32, shuffle=false);
loss_type = Flux.mse;
loss(diffusion, x) = p_losses(diffusion, loss_type, x; to_device=to_device)
opt = Adam(0.001);

println("Calculating initial loss")
val_loss = 0.0
for x in val_data
    global val_loss
    val_loss += loss(diffusion, x)
end
val_loss /= length(val_data)
@printf("\nval loss: %.5f\n", val_loss)

mkpath(directory)
output_path = joinpath(directory, "diffusion.bson")
history_path = joinpath(directory, "history.json")
hyperparameters_path = joinpath(directory, "hyperparameters.json")

hyperparameters = Dict(
    "num_timesteps" => num_timesteps,
    "data_shape" => "$(diffusion.data_shape)",
    "denoise_fn" => "$(typeof(model).name.wrapper)",
    "parameters" => sum(length, Flux.params(model)),
    "loss_type" => "$loss_type",
    "d_hid" => d_hid,
)
open(hyperparameters_path, "w") do f
    JSON.print(f, hyperparameters)
end
println("saved hyperparameters to $hyperparameters_path")

println("Starting training")
start_time = time_ns()
opt_state = Flux.setup(opt, diffusion)
history = train!(loss, diffusion, train_data, opt_state, val_data; num_epochs=num_epochs)
end_time = time_ns() - start_time
println("\ndone training")
@printf "time taken: %.2fs\n" end_time / 1e9

### save results
diffusion = diffusion |> cpu
BSON.bson(output_path, Dict(:diffusion => diffusion))
println("saved model to $output_path")

open(history_path, "w") do f
    JSON.print(f, history)
end
println("saved history to $history_path")

### plot results
diffusion = diffusion |> cpu

canvas_train = plot(
    1:length(history["mean_batch_loss"]), history["mean_batch_loss"], label="mean batch loss",
    xlabel="epoch",
    ylabel="loss",
    legend=:right, # :best, :right
    ylims=(0, Inf),
    )
plot!(canvas_train, 1:length(history["val_loss"]), history["val_loss"], label="validation loss")
savefig(canvas_train, joinpath(directory, "history.png"))
display(canvas_train)

X0 = p_sample_loop(diffusion, 1000)
canvas_samples = scatter(X0[1, :], X0[2, :], alpha=0.5, label="",
    aspectratio=:equal,
    xlims=(-2, 2), ylims=(-2, 2),
)
savefig(canvas_samples, joinpath(directory, "samples.png"))
display(canvas_samples)

println("press enter to finish")
readline()
