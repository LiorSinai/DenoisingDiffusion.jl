using Flux: batch
using ProgressMeter
using Images

function load_images(directory::AbstractString)
    filenames = readdir(directory);
    get_pos(filename::String) = parse(Int, match(r"\d+", filename).match)
    sort!(filenames, lt=(s1, s2)-> get_pos(s1) < get_pos(s2));

    xs = Array{Float32, 3}[]
    @showprogress for filename in filenames
        img = load(joinpath(directory, filename));
        img_CHW = channelview(img)
        img_WHC = permutedims(img_CHW, (3, 2, 1))
        x = Float32.(img_WHC)
        push!(xs, x)
    end
    X = batch(xs);
    X
end