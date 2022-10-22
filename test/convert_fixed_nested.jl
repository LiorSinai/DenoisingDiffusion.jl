function transfer_weights!(src::Dense, dest::Dense)
    dest.weight .= src.weight
    dest.bias .= src.bias
end

function transfer_weights!(src::Conv, dest::Conv)
    dest.weight .= src.weight
    if (src.bias !== false)
        dest.bias .= src.bias
    end
end

function transfer_weights!(src::GroupNorm, dest::GroupNorm)
    @assert dest.G == src.G
    dest.β .= src.β
    dest.γ .= src.γ
end

function transfer_weights!(src::ConvEmbed, dest::ConvEmbed)
    transfer_weights!(src.embed_layers[2], dest.embed_layers[2])
    transfer_weights!(src.conv, dest.conv)
    transfer_weights!(src.norm, dest.norm)
end

function transfer_weights!(src::ResBlock, dest::ResBlock)
    transfer_weights!(src.in_layers, dest.in_layers)
    transfer_weights!(src.out_layers[1], dest.out_layers[1])
    transfer_weights!(src.out_layers[2], dest.out_layers[2])
    if (typeof(src.skip_transform) <: Conv)
        transfer_weights!(src.skip_transform, dest.skip_transform)
    end
end

function transfer_weights!(src::MultiheadAttention, dest::MultiheadAttention)
    transfer_weights!(src.to_qkv, dest.to_qkv)
    transfer_weights!(src.to_out, dest.to_out)
end

function transfer_weights!(src::SkipConnection, dest::SkipConnection)
    transfer_weights!(src.layers, dest.layers)
end

function transfer_weights!(src::Chain, dest::Chain)
    for (src_layer, dest_layer) in zip(src.layers, dest.layers)
        transfer_weights!(src_layer, dest_layer)
    end
end

function transfer_weights!(src::T, dest::T) where T
end

function transfer_weights!(unet_fixed::UNetFixed, unet::UNet)
    @assert unet.num_levels == 4
    ## downs
    transfer_weights!(unet_fixed.embed_layers, unet.embed_layers) ;
    transfer_weights!(unet_fixed.downs[1], unet.chain[:init]) ;
    transfer_weights!(unet_fixed.downs[2], unet.chain[:down_1]) ;
    transfer_weights!(unet_fixed.downs[3], unet.chain[:skip_1].layers[:downsample_1]) ;
    transfer_weights!(unet_fixed.downs[4], unet.chain[:skip_1].layers[:down_2]) ;
    transfer_weights!(unet_fixed.downs[5], unet.chain[:skip_1].layers[:skip_2].layers[:downsample_2]) ;
    transfer_weights!(unet_fixed.downs[6], unet.chain[:skip_1].layers[:skip_2].layers[:down_3]) ;
    transfer_weights!(unet_fixed.downs[7], unet.chain[:skip_1].layers[:skip_2].layers[:skip_3].layers[:down_4]) ;

    ## middle
    transfer_weights!(unet_fixed.middle[1], unet.chain[:skip_1].layers[:skip_2].layers[:skip_3].layers[:middle_1]) ;
    transfer_weights!(unet_fixed.middle[2], unet.chain[:skip_1].layers[:skip_2].layers[:skip_3].layers[:middle_attention]) ;
    transfer_weights!(unet_fixed.middle[3], unet.chain[:skip_1].layers[:skip_2].layers[:skip_3].layers[:middle_2]) ;

    ## ups 
    transfer_weights!(unet_fixed.ups[1], unet.chain[:skip_1].layers[:skip_2].layers[:up_3]) ;
    transfer_weights!(unet_fixed.ups[2], unet.chain[:skip_1].layers[:skip_2].layers[:upsample_3]) ;
    transfer_weights!(unet_fixed.ups[3], unet.chain[:skip_1].layers[:up_2]) ;
    transfer_weights!(unet_fixed.ups[4], unet.chain[:skip_1].layers[:upsample_2]) ;
    transfer_weights!(unet_fixed.ups[5], unet.chain[:up_1]) ;
    transfer_weights!(unet_fixed.ups[6], unet.chain[:final]) ;

    nothing
end

@testset "UNetFixed=>UNet" begin
    unet_fixed = UNetFixed(1, 8, 100; block_layer=ResBlock);
    unet = UNet(1, 8, 100; block_layer=ResBlock, channel_multipliers=(1, 2, 4));

    transfer_weights!(unet_fixed, unet)

    x = rand(Float32, 28, 28, 1, 2);
    t = rand(1:100, 2);
    emb = unet_fixed.embed_layers(t);

    ### UnetFixed forward path
    d1f = unet_fixed.downs[1](x) ;
    d2f = unet_fixed.downs[2](d1f, emb);
    ###### skip 1
    d3f = unet_fixed.downs[3](d2f);
    d4f = unet_fixed.downs[4](d3f, emb);
    #### skip 2
    d5f = unet_fixed.downs[5](d4f);
    d6f = unet_fixed.downs[6](d5f, emb);
    ## skip 3
    d7f = unet_fixed.downs[7](d6f);
    m1f = unet_fixed.middle[1](d7f, emb);
    m2f = unet_fixed.middle[2](m1f);
    m3f = unet_fixed.middle[3](m2f, emb);
    mf = m3f
    ## skip3
    u1f = unet_fixed.ups[1](cat(d6f, mf, dims=3), emb);
    u2f = unet_fixed.ups[2](u1f);
    #### skip 2
    u3f = unet_fixed.ups[3](cat(d4f, u2f, dims=3), emb);
    u4f = unet_fixed.ups[4](u3f);
    ###### skip 1
    u5f = unet_fixed.ups[5](cat(d2f, u4f, dims=3), emb);
    u6f = unet_fixed.ups[6](u5f);

    #### tests
    ## skip3
    skip_layers = unet.chain[:skip_1].layers[:skip_2].layers[:skip_3].layers
    d7 = skip_layers[1](d6f)
    @test d7 == d7f
    m1 = skip_layers[2](d7, emb)
    @test m1 == m1f
    m2 = skip_layers[3](m1)
    @test m2 == m2f
    m3 = skip_layers[4](m2, emb)
    @test m3 == m3f

    ## skip 2
    skip_layers = unet.chain[:skip_1].layers[:skip_2].layers
    d5 =  skip_layers[1](d4f); #:downsample_2
    @test d5 == d5f
    d6 =  skip_layers[2](d5, emb); #:down_3
    @test d6 == d6f
    m1 =  skip_layers[3](d6, emb); #:skip_3
    @test m1 == cat(d6f, mf, dims=3)
    u1 =  skip_layers[4](m1, emb); #:up_3
    @test u1 == u1f
    u2 = skip_layers[5](u1); #:upsample_3
    @test u2 == u2f

    ## skip 1
    skip_layers = unet.chain[:skip_1].layers
    d3 = skip_layers[1](d2f); # :downsample_1
    @test d3 == d3f
    d4 = skip_layers[2](d3, emb); # :down_2
    @test d4 == d4f
    u2 = skip_layers[3](d4, emb); # :skip_2
    @test u2 == cat(d4f, u2f, dims=3)
    u3 = skip_layers[4](u2, emb); # :up_2
    @test u3 == u3f
    u4 = skip_layers[5](u3); # :upsample_2
    @test u4 == u4f

    ## entire model
    hf = unet_fixed(x, t);
    h = unet(x, t);
    @test hf == u6f
    @test h == hf
end
