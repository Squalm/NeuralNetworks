"""
Update the parameters of the model using the gardients (∇) and the learning rate (η).
"""
function update_model_weights(parameters, ∇, η)
    L = trunc(Int, length(parameters) / 2)

    # update the parameters (weights and biases) for all the layers
    for l = 1:L
        parameters[string("W_", l)] = min.(max.(parameters[string("W_", l)] - η .* ∇[string("δW_", l)], -100), 100)
        parameters[string("b_", l)] = min.(max.(parameters[string("b_", l)] - η .* ∇[string("δb_", l)], -100), 100)

        # Debugging printlns
        #println(∇[string("δW_", l)][1:5])
        #println(parameters[string("W_", l)][1:5])
    end # for

    #==topW = 0.0
    topB = 0.0
    bottomW = 0.0
    bottomB = 0.0
    for l in 1:L
        # normalise between -10 and 10
        # Weights
        _topW = maximum(parameters[string("W_", l)])
        _bottomW = minimum(parameters[string("W_", l)])
        if _topW > topW
            topW = _topW
        end # if
        if _bottomW < bottomW
            bottomW = _bottomW
        end # if
        # biases
        _topB = maximum(parameters[string("b_", l)])
        _bottomB = minimum(parameters[string("b_", l)])
        if _topB > topB
            topB = _topB
        end # if
        if _bottomB < bottomB
            bottomB = _bottomB
        end # if
    end # for
    for l in 1:L
        parameters[string("W_", l)] = [p > 0 ? 100 * p/topW : - 100 * p/bottomW for p in parameters[string("W_", l)]]
        parameters[string("b_", l)] = [b > 0 ? 100 * b/topB : - 100 * b/bottomB for b in parameters[string("b_", l)]]
    end # for==#
    return parameters
end # function