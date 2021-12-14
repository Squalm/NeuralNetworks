"""
Check the accuracy between predicted values (Ŷ) and the true value (Y).
"""
function assess_accuracy(Ŷ, Y)
    @assert size(Ŷ)[1] == size(Y)[1]
    return Ŷ[findmax(Y)[2]]
end # function