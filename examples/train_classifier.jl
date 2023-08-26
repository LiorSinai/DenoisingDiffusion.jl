using Flux
using Flux: DataLoader, onehotbatch, onecold
using Flux.Zygote: sensitivity, pullback
using BSON, JSON
using StatsBase: mean
using DenoisingDiffusion: split_validation, batched_metric, train!
using ProgressMeter, Printf
using Plots
using MLDatasets
using Random

include("LeNet5.jl")

#### settings
seed = 2714
data_directory = "path\\to\\data"
output_dir = "outputs\\LeNet5"
mkpath(output_dir)

### data
trainset = MNIST(Float32, :train, dir=data_directory)
testset = MNIST(Float32, :test, dir=data_directory)

X = reshape(trainset.features, 28, 28, 1, :)
y = onehotbatch(trainset.targets, 0:9)
X_test = reshape(testset.features, 28, 28, 1, :)
y_test = onehotbatch(testset.targets, 0:9);

rng = MersenneTwister(seed)
train_x, val_x = split_validation(rng, X, y; frac=0.1)
println("train data:      ", size(train_x[1]), ", ", size(train_x[2]))
println("validation data: ", size(val_x[1]), ", ", size(val_x[2]))

train_loader = Flux.DataLoader(train_x; batchsize=32, shuffle=true)
val_loader = Flux.DataLoader(val_x; batchsize=32, shuffle=false)
test_loader = Flux.DataLoader((X_test, y_test); batchsize=32, shuffle=false)

### build model
model = LeNet5()
display(model)
println("")

### definitions
accuracy(ŷ::AbstractMatrix, y::AbstractMatrix) = mean(onecold(ŷ, 0:9) .== onecold(y, 0:9))
loss(model, xy::Tuple) = Flux.logitcrossentropy(model(xy[1]), xy[2])

test_acc = batched_metric(model, accuracy, test_loader)
val_loss = 0.0
for x in val_loader
    val_loss += loss(model, x)
end
val_loss /= length(val_loader)
@printf("test accuracy for %d samples: %.2f%%\n", length(test_loader), test_acc * 100)
@printf("mean val loss: %.4f\n", val_loss)

println("training")
start_time = time_ns()
## The DenoisingDiffusion.train! method is meant for diffusion models but we can use it here.
## Ideally at the end of each epoch one should calculate additional metrics like train loss and accuracies.
## However with diffusion models these are expensive operations and so they are skipped.
opt_state = Flux.setup(Adam(0.001), model)
history = train!(
    loss, model, train_loader, opt_state, val_loader
    ; num_epochs=10,
    )
end_time = time_ns() - start_time
println("done training")
@printf "time taken: %.2fs\n" end_time / 1e9

### save
output_path = joinpath(output_dir, "model.bson")
history_path = joinpath(output_dir, "history.json")

BSON.bson(output_path, Dict(:model => model))
open(history_path, "w") do f
    JSON.print(f, history)
end

### test
test_acc = batched_metric(model, accuracy, test_loader)
@printf("test accuracy for %d samples: %.2f%%\n", length(test_loader), test_acc * 100)

epochs = 1:length(history["mean_batch_loss"])
canvas_loss = plot(
    epochs, history["mean_batch_loss"], label="mean batch loss",
    xlabel="epoch",
    ylabel="loss",
    legend=:right, # :best, :right
    ylims=(0, Inf),
)
plot!(canvas_loss, epochs, history["val_loss"], label="validation")

println("press enter to finish")
readline()