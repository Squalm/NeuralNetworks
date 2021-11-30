include("backward_linear_steps.jl")

"""
compute the gradients (∇) of the parameters (master_cache) of the constructed model with respect to the cost of prediciton (Ŷ) in comparison with actual output (Y).
"""
function back_propagate_model_weights(Ŷ, Y, master_cache)
    # initiate the dictionary to store the gradients for all the components in each layer
    ∇ = Dict()

    L = length(master_cache)
    Y = reshape(Y, size(Ŷ))

    # Partial derivate of the output layer
    δŶ = - (Y / Ŷ) + ((1 .- Y) / (1 .- Ŷ))
    current_cache = master_cache[L]

    # backpropagate on the layer preceeding the output layer
    ∇[string("δW_", (L))], ∇[string("δb_", (L))], ∇[string("δA_", (L-1))] = linear_activation_backwards(δŶ, current_cache, activation_function = "sigmoid")
    
    # go backward in the layers and compute the partial derivates of each component
    for l = reverse(0:L-2)
        current_cache = master_cache[l+1]
        ∇[string("δW_", (l+1))], ∇[string("δb_", (l+1))], ∇[string("δA_", (l))] = linear_activation_backwards(∇[string("δA_", (l+1))], current_cache, activation_function="relu")
    end # for

    return ∇
end # function
