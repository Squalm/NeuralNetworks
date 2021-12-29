using StableRNGs

"""
Function to initialise the parameters of the desired network.
"""
function initialise_model(layer_dims::Vector{Int}, seed::Any)
    params = Dict()

    for l = 2:length(layer_dims)
        params[string("W_", (l-1))] = rand( StableRNG(seed), layer_dims[l], layer_dims[l-1] ) * sqrt(2 / layer_dims[l-1])
        params[string("b_", (l-1))] = zeros(layer_dims[l], 1)
    end # for

    return params
end