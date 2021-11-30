include("activation_functions.jl")

"""
Derivative of Sigmoid
"""
function sigmoid_backwards(δA, activated_cache)
    s = sigmoid(activated_cache).A
    δZ = [δA[i] * s[i] * (1 - s[i]) for i in 1:length(s)]

    #δZ = [log(abs( 1 / x - 1 )) for x in δA]
    return δZ
end # function

"""
Derivative of ReLU
"""
function relu_backwards(δA, activated_cache)
    return [δA[i] * (activated_cache[i] > 0) for i in 1:length(δA)]
end # function
