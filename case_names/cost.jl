"""
Binary Cross Entropy calc.
"""
function calculate_cost(Ŷ, Y)
    m = size(Y, 2)
    epsilon = eps(1.0)

    # Deal with log(0)
    Ŷ_new = [min(i, epsilon) for i in [max(i, epsilon) for i in Ŷ]]

    # using hardcoded 29 as number of characters
    #cost = -sum(Y .* log2.(Ŷ_new) + ((1 + 2/29) .- Y) .* log2.((1 + 2/29) .- Ŷ_new)) /m
    cost = -sum([ Y[i] == 1 ? log2(abs(Ŷ_new[i])) : log2(abs(Ŷ_new[i] + 27/29)) for i in 1:length(Y)]) /m
    #cost = -log2(Ŷ_new[findmax(Y)[2]])
    return cost
end # function