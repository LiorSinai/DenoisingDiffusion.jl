@testset "DDIM" begin
    num_timesteps = 100
    model = UNet(1, 4, num_timesteps;
        block_layer=ResBlock,
        num_blocks_per_level=1,
        block_groups=1,
        channel_multipliers=(1, 2),
        num_attention_heads=4
    ) # 10_525 parameters
    βs = cosine_beta_schedule(num_timesteps)
    diffusion = GaussianDiffusion(Vector{Float32}, βs, (8, 8, 1,), model)

    nsamples = 3
    shape = (8, 8, 1, nsamples)
    x = randn(Float32, shape)
    noise = randn(Float32, size(x))
    timesteps = fill(100, nsamples)
    timesteps_next = fill(90, nsamples)
    x_prev, x_start = ddim_sample(diffusion, x, timesteps, timesteps_next, noise; η=1.0f0);
    @test size(x_prev) == (8, 8, 1, nsamples)
    @test size(x_start) == (8, 8, 1, nsamples)

    X0s = ddim_sample_loop(diffusion, 10, nsamples; to_device=cpu, η=1.0f0);
    @test size(X0s) == (8, 8, 1, nsamples)
end

@testset "DDIM - guidance" begin
    num_timesteps = 100
    model = UNetConditioned(1, 4, num_timesteps;
        num_classes=5,
        block_layer=ResBlock,
        num_blocks_per_level=1,
        block_groups=1,
        channel_multipliers=(1, 2),
        num_attention_heads=4
    ) # 11_197 parameters
    βs = cosine_beta_schedule(num_timesteps)
    diffusion = GaussianDiffusion(Vector{Float32}, βs, (8, 8, 1,), model)

    nsamples = 3
    shape = (8, 8, 1, nsamples)
    x = randn(Float32, shape)
    noise = randn(Float32, size(x))
    timesteps = fill(100, nsamples)
    timesteps_next = fill(90, nsamples)
    labels = rand(1:5, nsamples)
    x_prev, x_start = ddim_sample(diffusion, x, timesteps, timesteps_next, labels, noise; η=1.0f0);
    @test size(x_prev) == (8, 8, 1, nsamples)
    @test size(x_start) == (8, 8, 1, nsamples)

    X0s = ddim_sample_loop(diffusion, 10, labels; η=1.0f0);
    @test size(X0s) == (8, 8, 1, nsamples)

    label = 4
    X0s = ddim_sample_loop(diffusion, 10, nsamples, label; η=1.0f0);
    @test size(X0s) == (8, 8, 1, nsamples)
end