using Flux
using Flux: DataLoader, onehotbatch, onecold
using Flux.Zygote: sensitivity, pullback
using BSON, JSON
using StatsBase: mean
using DenoisingDiffusion: split_validation, count_observations, batched_metric
using ProgressMeter, Printf
using Plots
using MLDatasets
using Random

include("LeNet5.jl")

#### settings

seed = 2714
data_directory = "path\\to\\MNIST"
output_dir= "outputs\\LeNet5"

### data

trainset = MNIST(Float32, :train, dir=data_directory)
testset = MNIST(Float32, :test, dir=data_directory)

X = reshape(trainset.features, 28, 28, 1, :)
y = onehotbatch(trainset.targets, 0:9)
X_test = reshape(testset.features, 28, 28, 1, :)
y_test = onehotbatch(testset.targets, 0:9);

train_x, val_x = split_validation(MersenneTwister(seed), X, y; frac=0.1)
println("train data:      ", size(train_x[1]), ", ", size(train_x[2]))
println("validation data: ", size(val_x[1]), ", ", size(val_x[2]))

train_loader = Flux.DataLoader(train_x; batchsize=32, shuffle=true)
val_loader = Flux.DataLoader(val_x; batchsize=32, shuffle=false)
test_loader = Flux.DataLoader((X_test, y_test); batchsize=32, shuffle=false)

# build model

model = LeNet5()
display(model)
println("")

# definitions
accuracy(ŷ, y) = mean(onecold(ŷ, 0:9) .== onecold(y, 0:9))
loss(x, y) = Flux.logitcrossentropy(model(x), y)
loss(x::Tuple) = loss(x[1], x[2])
Flux.train!
opt = ADAM(0.001)

test_acc = batched_metric(accuracy, test_loader, model)
@printf("test accuracy for %d samples: %.2f%%\n", count_observations(test_loader), test_acc*100)

# custom training loop edited from Flux.jl/src/optimise/train.jl
function train!(loss, model, data, opt, val_data, accuracy; num_epochs=10)
    history = Dict(
        "train_acc"=>Float64[], 
        "val_acc"=>Float64[],
        "train_loss"=>Float64[], 
        "val_loss"=>Float64[], 
        )
    for epoch in 1:num_epochs
        losses = Vector{Float64}()
        params = Flux.params(model)
        progress = Progress(length(data); desc="epoch $epoch/$num_epochs")
        for x in data
            batch_loss, back = pullback(params) do
                loss(x)
            end    
            grads = back(sensitivity(batch_loss))
            Flux.update!(opt, params, grads)
            push!(losses, batch_loss)
            ProgressMeter.next!(progress; showvalues = [("batch loss", @sprintf("%.5f", batch_loss))])
        end
        # update history
        train_acc = batched_metric(accuracy, train_loader, model)
        valid_acc = batched_metric(accuracy, val_loader, model)

        push!(history["train_acc"], train_acc)
        push!(history["val_acc"], valid_acc)
        push!(history["train_loss"], mean(losses))
        push!(history["val_loss"], batched_metric(loss, val_data))

        @printf("train loss: %.5f ; ", history["train_loss"][end])
        @printf("val loss: %.5f ; ", history["val_loss"][end])
        @printf("train acc: %.2f%% ; ", history["train_acc"][end] * 100)
        @printf("val acc: %.2f%% ; ", history["val_acc"][end] * 100)
        println("")
    end
    history
end
println("training")
start_time = time_ns()
history = train!(loss, model, train_loader, opt, val_loader, accuracy; num_epochs=10)
end_time = time_ns() - start_time
println("done training")
@printf "time taken: %.2fs\n" end_time/1e9

## save
output_path = joinpath(output_dir, "model.bson")
history_path = joinpath(output_dir, "history.json")

BSON.bson(output_path, Dict(:model => model))
open(history_path, "w") do f
    JSON.print(f, history)
end

## test

test_acc = batched_metric(accuracy, test_loader, model)
@printf("test accuracy for %d samples: %.2f%%\n", count_observations(test_loader), test_acc*100)

epochs = 1:length(history["train_acc"])
canvas_acc = plot(
    epochs, history["train_acc"], label="train",
    xlabel="epochs",
    ylabel="accuracy",
    legend=:right, # :best, :right
    ylims=(0, 1),
    )
plot!(canvas_acc, epochs, history["val_acc"], label="valid")
plot!(canvas_acc, [epochs[end]], [test_acc], markershape=:star, label="test")

epochs = 1:length(history["train_loss"])
canvas_loss = plot(
    epochs, history["train_loss"], label="train",
    xlabel="epochs",
    ylabel="loss",
    legend=:right, # :best, :right
    ylims=(0, Inf),
    )
plot!(canvas_loss, epochs, history["val_loss"], label="valid")

plot(canvas_loss, canvas_acc, plot_title="history", size=(800, 400), margin=3Plots.mm)