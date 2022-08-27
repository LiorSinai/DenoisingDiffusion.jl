
@testset "ConvEmbed" begin
    model = ConvEmbed(1=>8, 16; groups=8);
    x = rand(Float32, 28, 28, 1, 2);
    t = rand(Float32, 16, 2)
    @test_nowarn model(x, t)
end

@testset "ResBlock" begin
    model = ResBlock(1=>8, 16; groups=8);
    x = rand(Float32, 28, 28, 1, 2);
    t = rand(Float32, 16, 2)
    @test_nowarn model(x, t)
end

@testset "UnetFixed" begin
    model = UNetFixed(1, 8, 10; block_layer=ConvEmbed);
    x1 = rand(Float32, 28, 28, 1, 2);
    #x2 = rand(Float32, 27, 27, 1, 2); # doesn't work for odd numbers
    t1 = rand(1:10, 2)
    t2 = rand(1:10, 3)

    @test_nowarn model(x1, t1)
    output = model(x1, t1)
    @test size(output) == size(x1)

    @test_throws DimensionMismatch model(x1, t2)
    @test_throws Exception UNetFixed(1, 8, 10; block_groups=32);

    model = UNetFixed(1, 8, 10; block_layer=ResBlock)
    @test_nowarn model(x1, t1)
end

@testset "Unet" begin
    x = rand(Float32, 32, 32, 1, 2);
    t = rand(1:10, 2)

    model = UNet(1, 8, 10; block_layer=ConvEmbed, channel_multipliers=(1, 2, 4));

    @test_nowarn model(x, t)
    output = model(x, t)
    @test size(output) == size(x)

    model = UNet(1, 8, 10; block_layer=ConvEmbed, channel_multipliers=(1, 2, 4, 8));
    @test_nowarn model(x, t)

    model = UNet(1, 8, 10; block_layer=ConvEmbed, channel_multipliers=(1, 3, 3, 4));
    @test_nowarn model(x, t)
end
