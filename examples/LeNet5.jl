#=
LeNet 5

layers:     8
parameters: 44,426
size:       174.867 KiB
source:     https://github.com/FluxML/model-zoo/blob/master/vision/conv_mnist/conv_mnist.jl
            https://d2l.ai/chapter_convolutional-neural-networks/lenet.html 
=#

function LeNet5(; imgsize=(28, 28, 1), nclasses=10) 
    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    
    return Chain(
            Conv((5, 5), imgsize[end]=>6, relu),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2, 2)),
            Flux.flatten,
            Dense(prod(out_conv_size), 120, relu), 
            Dense(120, 84, relu), 
            Dense(84, nclasses)
          )
end