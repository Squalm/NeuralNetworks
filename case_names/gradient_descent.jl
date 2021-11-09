"""
Update the parameters of the model using the gardients (∇) and the learning rate (η).
"""
function update_model_weights(parameters, ∇, η)
    L = floor(length(parameters) / 2)

    # update the parameters (weights and biases) for all the layers
    for l = 0:trunc(Int, L)-1
        parameters[string("W_", (l+1))] -= η .* ∇[string("δW_", (l+1))]
        parameters[string("b_", (l+1))] -= η .* ∇[string("δb_", (l+1))]
    end # for
    
    return parameters
end # function