"""
Sigmoid Activation function
"""
function sigmoid(Z::Array{BigFloat})
    A = max.(min.(1 ./ (1 .+ exp.(.-Z)), 1-eps(1.0)), eps(1.0))
    return (A = A, Z = Z)
end # function

"""
ReLU Activation function
"""
function relu(Z)
    A = max.(0, Z)
    return (A = A, Z = Z)
end # function

"""
Softmax Activation function
"""
function softmax(Z::Array{BigFloat})
    A = max.(exp.(Z) ./ ( sum(exp.(Z))), eps(1.0))
    return (A = A, Z = Z)
end # function

"""
Tanh Activation function
"""
function tanhact(Z::Array{BigFloat})
    A = min.(max.(tanh.(Z), -1.0+eps(1.0)), 1.0-eps(1.0))
    return (A = A, Z = Z)
end # function

"""
Swish Activation function
"""
function swish(Z::Array{BigFloat})
    #Z₁ = sigmoid(Z).A
    #A = [Z[i] * Z₁[i] for i in 1:length(Z)]
    A = Z .* sigmoid(Z).A
    return (A = A, Z = Z)
end # function

"""
GELU Activation function
"""
function gelu(Z)
    A = 0.5 * Z * (1 + tanh( sqrt(pi / 2) * (Z + 0.044715 * Z^3) ))
    return (A = A, Z = Z)
end # function