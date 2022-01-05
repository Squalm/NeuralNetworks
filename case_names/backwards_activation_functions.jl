include("activation_functions.jl")

"""
Derivative of Sigmoid
"""
function sigmoid_backwards(δA, activated_cache)
    s = sigmoid(activated_cache).A
    s = [max(min(x, 1-(eps(1.0))), eps(1.0)) for x in s]
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

"""
Derivative of Tanh
"""
function tanh_backwards(δA, activated_cache)
    return [δA[i] * (1 - (tanh(activated_cache[i]))^2) for i in 1:length(δA)]
end # function

"""
Derivative of Softmax
"""
function softmax_backwards(δA, activated_cache)
    return δA * ( - exp.(activated_cache) .* sum(exp.(activated_cache)) )
end # function

"""
Derivative of Swish
"""
function swish_backwards(δA, activated_cache)
    # β = 1
    return δA * swish(activated_cache).A + sigmoid(activated_cache).A * (1 .- swish(activated_cache).A)
end # function