using ProgressMeter
using Images

function load_images(directory::AbstractString)
    filenames = readdir(directory);
    get_pos(filename::String) = parse(Int, match(r"\d+", filename).match)
    sort!(filenames, lt=(s1, s2)-> get_pos(s1) < get_pos(s2));

    img1 = load(joinpath(directory, filenames[1]))
    img1_WHC = convert_width_height_channel(img1)
    nbatch = length(filenames)
    X = Array{Float32}(undef, size(img1_WHC)..., nbatch)
    @showprogress for (idx, filename) in enumerate(filenames)
        img = load(joinpath(directory, filename));
        X[:, :, :, idx] = convert_width_height_channel(img)
    end
    X
end

function convert_width_height_channel(img::AbstractMatrix)
    img_CHW = channelview(img)
    img_WHC = permutedims(img_CHW, (3, 2, 1))
    img_WHC
end