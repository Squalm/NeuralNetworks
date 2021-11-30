"""
Update the parameters of the model using the gardients (∇) and the learning rate (η).
"""
function update_model_weights(parameters, ∇, η::Float)
    L = trunc(Int, length(parameters) / 2)

    # update the parameters (weights and biases) for all the layers
    for l = 1:L
        parameters[string("W_", l)] -= η .* ∇[string("δW_", l)]
        parameters[string("b_", l)] -= η .* ∇[string("δb_", l)]
    end # for

    return parameters
end # function