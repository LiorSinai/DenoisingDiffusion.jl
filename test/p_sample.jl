@testset "p_sample" begin
    num_timesteps = 100
    model = ConditionalChain(
        Parallel(
            .+,
            Dense(2, 16),
            Embedding(num_timesteps => 16),
        ),
        Dense(16 => 2)
    )
    βs = cosine_beta_schedule(num_timesteps)
    diffusion = GaussianDiffusion(Vector{Float32}, βs, (2,), model)

    nsamples = 3
    shape = (2, nsamples)
    x = randn(Float32, shape)
    timesteps = rand(1:num_timesteps, nsamples)
    noise = randn(Float32, size(x))
    x_prev, x_start = p_sample(diffusion, x, timesteps, noise);
    @test size(x_prev) == (2, nsamples)
    @test size(x_start) == (2, nsamples)

    Xs = p_sample_loop(diffusion, 7);
    @test size(Xs) == (2, 7)
end

@testset "p_sample - guidance" begin
    num_timesteps = 100
    model = ConditionalChain(
        Parallel(
            .+,
            Dense(2, 16),
            Embedding(num_timesteps => 16),
            Embedding(5 => 16),
        ),
        Dense(16 => 2),
    )
    βs = cosine_beta_schedule(num_timesteps)
    diffusion = GaussianDiffusion(Vector{Float32}, βs, (2,), model)

    nsamples = 3
    shape = (2, nsamples)
    x = randn(Float32, shape)
    timesteps = rand(1:num_timesteps, nsamples)
    noise = randn(Float32, size(x))
    labels = rand(1:5, nsamples)
    x_prev, x_start = p_sample(diffusion, x, timesteps, labels, noise);
    @test size(x_prev) == (2, nsamples)
    @test size(x_start) == (2, nsamples)

    Xs = p_sample_loop(diffusion, labels);
    @test size(Xs) == (2, nsamples)
end