include("backwards_activation_functions.jl")

"""
Partial derivates of the components of linear forward function using the linear output (δZ) and caches of these components (cache).
"""
function linear_backward(δZ, cache)
    # unpack cache
    A_prev, W, b = cache
    m = size(A_prev, 2)

    # partial derivates of each component
    δW = δZ * (A_prev') / m
    δb = sum(δZ, dims = 2) / m
    δA_prev = (W') * δZ

    @assert (size(δA_prev) == size(A_prev))
    @assert (size(δW) == size(W))
    @assert (size(δb) == size(b))

    return δW, δb, δA_prev
end # function

"""
Unpack the linear activated caches (cache) and compute their derivates from the applied activation function.
"""
function linear_activation_backwards(δA, cache; activation_function="relu")
    @assert activation_function ∈ ("sigmoiod", "relu")

    linear_cache, cache_activation = cache

    if (activation_function == "relu")

        δZ = relu_backwards(δA, cache_activation)
        δW, δb, δA_prev = linear_backward(δZ, linear_cache)

    elseif (activation_function == "sigmoid")

        δZ = sigmoid_backwards(δA, cache_activation)
        δW, δb, δA_prev = linear_backward(δZ, linear_cache)

    end # if

    return δW, δb, δA_prev
end # function