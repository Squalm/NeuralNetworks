"""
Sigmoid activation function
"""
function sigmoid(Z)
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