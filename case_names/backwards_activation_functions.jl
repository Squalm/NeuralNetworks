include("activation_functions.jl")

"""
Derivate of Sigmoid
"""
function sigmoid_backwards(δA, activated_cache)
    s = sigmoid(activated_cache).A
    δZ = δA .* s .* (1 .- s)

    @assert (size(δZ) == size(activated_cache))

    return δZ
end # function

"""
Derivate of ReLU
"""
function relu_backwards(δA, activated_cache)
    return δA .* (activated_cache .> 0)
end # function
