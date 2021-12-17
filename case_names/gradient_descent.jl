"""
Update the parameters of the model using the gardients (∇) and the learning rate (η).
"""
function update_model_weights(parameters, ∇, η)
    L = trunc(Int, length(parameters) / 2)

    # update the parameters (weights and biases) for all the layers
    for l = 1:L
        parameters[string("W_", l)] = max.(min.(parameters[string("W_", l)] - η .* ∇[string("δW_", l)], 100), -100)
        parameters[string("b_", l)] = max.(min.(parameters[string("b_", l)] - η .* ∇[string("δb_", l)], 1000), -1000)
        # Debugging printlns
        #println(∇[string("δW_", l)][1:5])
        #println(parameters[string("W_", l)][1:5])
    end # for

    return parameters
end # function