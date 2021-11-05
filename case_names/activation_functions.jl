"""
Sigmoid activation function
"""
function sigmoid(Z)
    A = 1 ./ (1 .+ exp.(.-Z))
    return (A = A, Z = Z)
end # function

"""
ReLU Activation function
"""
function relu(Z)
    A = max.(0, Z)
    return (A = A, Z = Z)
end # function