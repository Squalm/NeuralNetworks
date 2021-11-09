"""
Check the accuracy between predicted values (Ŷ) and the true value (Y).
"""
function assess_accuracy(Ŷ, Y)
    @assert size(Ŷ)[1] == size(Y)[1]
    return sum((Ŷ .> 0.5) .== Y) / length(Y)
end # function