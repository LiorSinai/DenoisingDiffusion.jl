using Random
using DenoisingDiffusion: split_validation

@testset "split_validation" begin
    data = 1:10
    train_data, val_data = split_validation(MersenneTwister(0), data; frac=0.2)
    @test issetequal(val_data, [1, 9])
    @test issetequal(train_data, [7, 4, 6, 8, 2, 3, 5, 10])

    # these are shared in all the following tests
    val_idxs = [12, 19, 5, 15]
    train_idxs = [7, 4, 6, 8, 20, 16, 18, 17, 9, 1, 14, 13, 10, 3, 2, 11]

    data = rand(32, 32, 20)
    train_data, val_data = split_validation(MersenneTwister(0), data; frac=0.2)
    @test val_data == data[:, :, val_idxs]
    @test train_data == data[:, :, train_idxs]

    data = rand(32, 32, 3, 20)
    train_data, val_data = split_validation(MersenneTwister(0), data; frac=0.2)
    @test val_data == data[:, :, :, val_idxs]
    @test train_data == data[:, :, :, train_idxs]

    labels = rand(1:4, 20)
    data = rand(32, 32, 3, 20)
    train_data, val_data = split_validation(MersenneTwister(0), data, labels; frac=0.2)
    @test val_data[1] == data[:, :, :, val_idxs]
    @test train_data[1] == data[:, :, :, train_idxs]
    @test val_data[2] == labels[val_idxs]
    @test train_data[2] == labels[train_idxs]

    labels = rand(1:4, 20)
    onehotlabels = zeros(Int, 4, 20)
    for (i, l) in enumerate(labels)
        onehotlabels[l, i] = 1
    end
    data = rand(32, 32, 3, 20)
    train_data, val_data = split_validation(MersenneTwister(0), data, onehotlabels; frac=0.2)
    @test val_data[1] == data[:, :, :, val_idxs]
    @test train_data[1] == data[:, :, :, train_idxs]
    @test val_data[2] == onehotlabels[:, val_idxs]
    @test train_data[2] == onehotlabels[:, train_idxs]
end