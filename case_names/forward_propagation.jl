include("linear_steps.jl")

"""
Iterate through the layers in the net to get a prediction.
"""
function forward_propagate_model_weights(DMatrix, parameters)
    master_cache = []
    A = DMatrix
    L = trunc(Int, length(parameters) / 2)

    # Forawrd propagate until the last layer
    for l = 1:(L-1)
        A_prev = A
        A, cache = linear_forward_activate(A_prev, parameters[string("W_", (l))], parameters[string("b_", (l))], activation_function = "tanh")
        # Debugging println statements
        if l == L-1
            println(string(l, " : ", A[1:2]))
        end # if
        push!(master_cache, cache)
    end # for

    # For the final layer (outputs)
    Ŷ, cache = linear_forward_activate(A, parameters[string("W_", (L))], parameters[string("b_", (L))], activation_function = "softmax")
    # Make sure the outputs are within bounds too
    # Debuggin println
    println(string(L, " : ", Ŷ[1:2]))
    push!(master_cache, cache)

    return Ŷ, master_cache
end # function