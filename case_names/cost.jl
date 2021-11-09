"""
Binary Cross Entropy calc.
"""
function calculate_cost(Ŷ, Y)
    m = size(Y, 2)
    epsilon = eps(1.0)

    # Deal with log(0)
    Ŷ_new = [max(i, epsilon) for i in Ŷ]
    Ŷ_new = [min(i, epsilon) for i in Ŷ_new]

    cost = -sum(Y .* log10.(Ŷ_new) + (ones(size(Y)) - Y) .* log10.(ones(size(Ŷ_new)) - Ŷ_new)) / m
    return cost
end # function