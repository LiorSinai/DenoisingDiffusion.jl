@testset "MultiheadAttention" begin
    model = MultiheadAttention(15, nhead=3);
    dh = div(15, 3)
    x = rand(Float32, 8, 8, 15, 2);
    @test_nowarn model(x)
end

@testset "Mutli same as single" begin
    dh = 5

    Q = rand(Float32, dh, 8*8, 3, 2)
    K = rand(Float32, dh, 8*8, 3, 2)
    V = rand(Float32, dh, 8*8, 3, 2)

    batch_idx = 1
    head_idx = 2
    q = Q[:, :, head_idx, batch_idx]
    k = K[:, :, head_idx, batch_idx]
    v = V[:, :, head_idx, batch_idx]

    attn_one = scaled_dot_attention(q, k, v)
    attn_multi = scaled_dot_attention(Q, K, V)

    @test attn_multi[:, :, head_idx, batch_idx] ≈ attn_one
end

