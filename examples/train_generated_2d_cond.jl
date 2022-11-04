using Plots
using Flux
using Flux: Embedding
using Dates
using BSON, JSON
using Printf

using Revise
using DenoisingDiffusion
using DenoisingDiffusion: train!
includet("datasets.jl")
includet("utilities.jl")

## settings

directory = "outputs\\" * Dates.format(now(), "yyyymmdd_HHMM")
num_timesteps = 40
n_batch = 9_000
to_device = cpu
d_hid = 32
num_epochs = 100
num_classes = 3
p_uncond = 0.2

## data
nsamples_per_class = round(Int, n_batch / num_classes)
X1 = normalize_neg_one_to_one(make_spiral(nsamples_per_class));
X2 = normalize_neg_one_to_one(make_s_curve(nsamples_per_class));
X3 = normalize_neg_one_to_one(make_moons(nsamples_per_class));

X = hcat(X1, X2, X3)
labels = 1 .+ vcat(fill(1, nsamples_per_class), fill(2, nsamples_per_class), fill(3, nsamples_per_class))

n_val = floor(Int, 0.1 * nsamples_per_class)
X1_val = normalize_neg_one_to_one(make_spiral(n_val));
X2_val = normalize_neg_one_to_one(make_s_curve(n_val));
X3_val = normalize_neg_one_to_one(make_moons(n_val));

X_val = hcat(X1_val, X2_val, X3_val)
labels_val = 1 .+ vcat(fill(1, n_val), fill(2, n_val), fill(3, n_val))

## model
model = ConditionalChain(
    Parallel(
        .+, Dense(2, d_hid), 
        Chain(SinusoidalPositionEmbedding(num_timesteps, d_hid), 
        Dense(d_hid, d_hid)), 
        Embedding(1 + num_classes => d_hid)
    ),
    swish,
    Parallel(
        .+, 
        Dense(d_hid, d_hid), 
        Chain(SinusoidalPositionEmbedding(num_timesteps, d_hid), Dense(d_hid, d_hid)), 
        Embedding(1 + num_classes => d_hid)
    ),
    swish,
    Parallel(
        .+, 
        Dense(d_hid, d_hid), 
        Chain(SinusoidalPositionEmbedding(num_timesteps, d_hid), 
        Dense(d_hid, d_hid)), 
        Embedding(1 + num_classes => d_hid)
        ),
    swish,
    Dense(d_hid, 2),
)
display(model)

βs = linear_beta_schedule(num_timesteps, 8e-6, 9e-5)
diffusion = GaussianDiffusion(Vector{Float32}, βs, (2,), model)

### train
diffusion = diffusion |> to_device

data = Flux.DataLoader((X, labels) |> to_device; batchsize=32, shuffle=true);
val_data = Flux.DataLoader((X_val, labels_val) |> to_device; batchsize=32, shuffle=false);
loss_type = Flux.mse;
loss(diffusion, x) = p_losses(diffusion, loss_type, x; to_device=to_device, p_uncond=p_uncond)
opt = Adam(0.001);

println("Calculating initial loss")
val_loss = 0.0
for x in val_data
    global val_loss
    val_loss += loss(diffusion, x)
end
val_loss /= length(val_data)
@printf("\nval loss: %.5f\n", val_loss)


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
println("saved model to $output_path")

open(history_path, "w") do f
    JSON.print(f, history)
end
println("saved history to $history_path")

hyperparameters = Dict(
    "num_timesteps" => num_timesteps,
    "data_shape" => "$(diffusion.data_shape)",
    "denoise_fn" => "$(typeof(model).name.wrapper)",
    "parameters" => sum(length, Flux.params(model)),
    "loss_type" => "$loss_type",
    "d_hid" => d_hid,
    "num_classes" => num_classes,
    "p_uncond" => p_uncond,
)

open(hyperparameters_path, "w") do f
    JSON.print(f, hyperparameters)
end
println("saved hyperparameters to $hyperparameters_path")

### plot results
diffusion = diffusion |> cpu

p = plot(1:length(history["train_loss"]), history["train_loss"], label="train_loss")
plot!(p, 1:length(history["val_loss"]), history["val_loss"], label="val_loss")
display(p)
canvases = []
for label in 1:4
    X0 = p_sample_loop(diffusion, 1000, label; guidance_scale=1.0f0)
    p0 = scatter(X0[1, :], X0[2, :], alpha=0.5, label="",
        aspectratio=:equal,
        xlims=(-2, 2), ylims=(-2, 2),
        title="label=$label"
    )
    push!(canvases, p0)
end
p = plot(canvases...)
display(p)

println("press enter to finish")
readline()
