using MLDatasets
using Plots, Images
using Flux
using Dates
using BSON, JSON
using Printf
using Random
using ProgressMeter

using Revise
using DenoisingDiffusion
using DenoisingDiffusion: train!, split_validation, load_opt_state!
include("utilities.jl")

### settings

num_timesteps = 100
seed = 2714
dataset = :MNIST
data_directory = "path\\to\\MNIST"
output_directory = "outputs\\$(dataset)_" * Dates.format(now(), "yyyymmdd_HHMM")
model_channels = 16
learning_rate = 0.001
combine_embeddings = vcat
num_epochs = 10
p_uncond = 0.2
loss_type = Flux.mse;
to_device = gpu # cpu or gpu
num_classes = 10

### data

trainset = MNIST(Float32, :train, dir=data_directory);
norm_data = normalize_neg_one_to_one(reshape(trainset.features, 28, 28, 1, :));
labels = 2 .+ trainset.targets; # 1->default, 2->0, 3->1, ..
train_x, val_x = split_validation(MersenneTwister(seed), norm_data, labels);

println("train data:      ", size(train_x[1]), "--", size(train_x[2]))
println("validation data: ", size(val_x[1]), "--",  size(val_x[2]))

### model
## create
in_channels = size(train_x[1], 3)
data_shape = size(train_x[1])[1:3]
model = UNetConditioned(in_channels, model_channels, num_timesteps; 
    num_classes=num_classes,
    block_layer=ResBlock, block_groups=8, channel_multipliers=(1, 2, 4), 
    num_attention_heads=4, combine_embeddings=combine_embeddings
    )
βs = cosine_beta_schedule(num_timesteps, 0.008)
diffusion = GaussianDiffusion(Vector{Float32}, βs, data_shape, model)
## load
# BSON.@load "outputs\\MNIST_20221031_16cond\\diffusion_opt.bson" diffusion opt
# params_start = Flux.params(diffusion);

display(diffusion.denoise_fn)
println("")

### train
diffusion = diffusion |> to_device

data = Flux.DataLoader(train_x |> to_device; batchsize=32, shuffle=true);
val_data = Flux.DataLoader(val_x |> to_device; batchsize=32, shuffle=false);
loss(diffusion, x) = p_losses(diffusion, loss_type, x; to_device=to_device, p_uncond=p_uncond)
if isdefined(Main, :opt)
    println("loading optimiser state")
    load_opt_state!(opt, params_start, Flux.params(diffusion), to_device=to_device)
    println("  length(opt.state) = ", length(opt.state))
else
    println("defining new optimiser")
    opt = Adam(learning_rate);
    println("  ", opt)
end

println("Calculating initial loss")
val_loss = 0.0
@showprogress for x in val_data
    global val_loss
    val_loss += loss(diffusion, x)
end
val_loss /= length(val_data)
@printf("\nval loss: %.5f\n", val_loss)

mkdir(output_directory)
println("made directory: ", output_directory)
hyperparameters_path = joinpath(output_directory, "hyperparameters.json")
output_path = joinpath(output_directory, "diffusion_opt.bson")
history_path = joinpath(output_directory, "history.json")

hyperparameters = Dict(
    "dataset" => "$dataset",
    "num_timesteps" => num_timesteps,
    "data_shape" => "$(diffusion.data_shape)",
    "denoise_fn" => "$(typeof(diffusion.denoise_fn).name.wrapper)",
    "combine_embeddings" => "$combine_embeddings",
    "parameters" => sum(length, Flux.params(diffusion.denoise_fn)),
    "model_channels" => model_channels,
    "seed" => seed,
    "loss_type" => "$loss_type",
    "learning_rate" => learning_rate,
    "p_uncond" => p_uncond,
    "num_classes" => num_classes,
    "optimiser" => "$(typeof(opt).name.wrapper)",
)
open(hyperparameters_path, "w") do f
    JSON.print(f, hyperparameters)
end
println("saved hyperparameters to $hyperparameters_path")

println("Starting training")
start_time = time_ns()
history = train!(loss, diffusion, data, opt, val_data; 
    num_epochs=num_epochs, save_after_epoch=true, save_dir=output_directory)
end_time = time_ns() - start_time
println("\ndone training")
@printf "time taken: %.2fs\n" end_time / 1e9

### save results

open(history_path, "w") do f
    JSON.print(f, history)
end
println("saved history to $history_path")

params_device = Flux.params(diffusion);
let diffusion = cpu(diffusion)
    # save opt in case want to resume training
    load_opt_state!(opt, params_device, Flux.params(diffusion), to_device=cpu)
    BSON.bson(output_path, Dict(:diffusion => diffusion, :opt => opt )) 
end
println("saved model to $output_path")

### plot results

p1 = plot(1:length(history["val_loss"]), history["val_loss"], label="val loss")
display(p1)

X0_all = p_sample_loop(diffusion, collect(1:11); guidance_scale=2.0f0, to_device=to_device);
X0_all = X0_all |> cpu ;
imgs = convert2image(trainset, X0_all[:, :, 1, :]);
p_all = plot([plot(imgs[:, :, i], title="digit=$(i-2)") for i in 1:11]..., ticks=nothing)
display(p_all)

for label in 1:11
    println("press enter for next label")
    readline()
    X0 = p_sample_loop(diffusion, 12, label; guidance_scale=2.0f0, to_device=to_device)
    X0 = X0 |> cpu
    imgs = convert2image(trainset, X0[:, :, 1, :])
    p0 = plot([plot(imgs[:, :, i]) for i in 1:12]..., plot_title ="label=$label", ticks=nothing)
    display(p0) 
end

println("press enter to finish")
readline()
