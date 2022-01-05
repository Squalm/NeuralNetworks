include("activation_functions.jl")

"""
Make a linear forward calc.
"""
function linear_forward(A, W, b)
    # Make a linear forward and return the inputs as cache
    Z = BigFloat.((W * A) .+ b)
    cache = (A, W, b)

    @assert size(Z) == (size(W, 1), size(A, 2))

    return (Z = Z, cache = cache)
end # function

"""
Make a forward activation
"""
function linear_forward_activate(A_prev, W, b; activation_function = "tanh")
    @assert activation_function ∈ ("sigmoid", "relu", "softmax", "tanh", "swish")
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation_function == "sigmoid"
        A, activation_cache = sigmoid(Z)
    end # if

    if activation_function == "relu"
        A, activation_cache = relu(Z)
    end # if

    if activation_function == "softmax"
        A, activation_cache = softmax(Z)
    end # if

    if activation_function == "tanh"
        A, activation_cache = tanhact(Z)
    end # if

    if activation_function == "swish"
        A, activation_cache = swish(Z)
    end # if

    cache = (linear_step_cache = linear_cache, activation_step_cache = activation_cache)

    @assert size(A)[1] == size(W, 1)[1]

    return A, cache
end # function
