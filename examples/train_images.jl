using MLDatasets
using Plots, Images
using Flux
using Dates
using BSON, JSON
using Printf
using Random

using DenoisingDiffusion
using DenoisingDiffusion: train!, split_validation, load_opt_state!
include("utilities.jl")

### settings

num_timesteps = 100
seed = 2714
dataset = :MNIST  # :MNIST or :Pokemon
data_directory = "path\\to\\MNIST"
output_directory = "outputs\\$(dataset)_" * Dates.format(now(), "yyyymmdd_HHMM")
model_channels = 16
learning_rate = 0.001
num_epochs = 10
loss_type = Flux.mse;
to_device = gpu # cpu or gpu

### data

if dataset == :MNIST
    trainset = MNIST(Float32, :train, dir=data_directory)
    norm_data = normalize_neg_one_to_one(reshape(trainset.features, 28, 28, 1, :))
    train_x, val_x = split_validation(MersenneTwister(seed), norm_data)
elseif dataset == :Pokemon
    data_path = joinpath(data_directory, "imgs_WHCN_48x48.bson")
    data = BSON.load(data_path)[:imgs_WHCN]; 
    norm_data = normalize_neg_one_to_one(data);
    train_x, val_test_x = split_validation(MersenneTwister(seed), norm_data);
    n_val = floor(Int, size(val_test_x, 4)/2)
    n_train = size(train_x, 4)
    val_x = val_test_x[:, :, :, 1:n_val]
    test_x = val_test_x[:, :, :, (n_val + 1):end]
else 
    throw("$dataset not supported")
end

println("train data:      ", size(train_x))
println("validation data: ", size(val_x))

### model
## create
in_channels = size(train_x, 3)
data_shape = size(train_x)[1:3]
model = UNet(in_channels, model_channels, num_timesteps; block_layer=ResBlock, block_groups=8, channel_multipliers=(1, 2, 4))
βs = cosine_beta_schedule(num_timesteps, 0.008)
diffusion = GaussianDiffusion(Vector{Float32}, βs, data_shape, model)
## load
# BSON.@load "outputs\\MNIST_20220814_2214\\diffusion_opt.bson" diffusion opt
# params_start = Flux.params(diffusion);

display(diffusion.denoise_fn)
println("")

### train
diffusion = diffusion |> to_device

data = Flux.DataLoader(train_x |> to_device; batchsize=32, shuffle=true);
val_data = Flux.DataLoader(val_x |> to_device; batchsize=32, shuffle=false);
loss(diffusion, x) = p_lossess(diffusion, loss_type, x; to_device=to_device)
if isdefined(Main, :opt)
    println("loading optimiser state")
    load_opt_state!(opt, params_start, Flux.params(diffusion), to_device=to_device)
    println("  length(opt.state) = ", length(opt.state))
else
    println("defining new optimiser")
    opt = Adam(learning_rate);
    println("  ", opt)
end

val_loss = 0.0
for x in val_data
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
    "parameters" => sum(length, Flux.params(diffusion.denoise_fn)),
    "model_channels" => model_channels,
    "seed" => seed,
    "loss_type" => "$loss_type",
    "learning_rate" => learning_rate,
    "optimiser" => "$(typeof(opt).name.wrapper)",
)
open(hyperparameters_path, "w") do f
    JSON.print(f, hyperparameters)
end

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

params_device = Flux.params(diffusion);
let diffusion = cpu(diffusion)
    # save opt in case want to resume training
    load_opt_state!(opt, params_device, Flux.params(diffusion), to_device=cpu)
    BSON.bson(output_path, Dict(:diffusion => diffusion, :opt => opt )) 
end

### plot results

p = plot(1:length(history["val_loss"]), history["val_loss"], label="val loss")
display(p)

X0 = p_sample_loop(diffusion, 12; to_device=to_device)
X0 = X0 |> cpu

if dataset == :MNIST
    imgs = convert2image(trainset, X0[:, :, 1, :])
elseif dataset == :Pokemon
    for i in 1:12
        X0[:, :, :, i] = normalize_zero_to_one(X0[:, :, :, i])
    end
    imgs = img_WHC_to_rgb(X0)
end

p = plot([plot(imgs[:, :, i]) for i in 1:12]...)
display(p)

println("press enter to finish")
readline()
