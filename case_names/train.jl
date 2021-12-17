using DataFrames, CSV

include("initialise.jl")
include("forward_propagation.jl")
include("cost.jl")
include("check_accuracy.jl")
include("backpropagation.jl")
include("gradient_descent.jl")

"""
Train the network using the desired architecture that best possible matches the training inputs (DMatrix) and their corresponding outputs(Y) over some number of iterations (epochs) and a learning rate (η).
"""
function train_network(layer_dims::Vector{Int64}, DMatrix::Vector{Any}, Y::Vector{Any}, mapping::String; η = 0.001, epochs = 500, seed = 424367, verbose = true)

    costs = []
    iters = []
    accuracy = []
    yn = []
    YvŶ = [[], []]
    mapping *= "#"

    # initialise
    params = initialise_model(layer_dims, seed)

    # LOGGING
    d = string(Dates.now())[1:13] * "-" * string(Dates.now())[15:16]

    # train
    for i = 1:epochs

        for datum in 1:length(DMatrix)
            Ŷ, caches = forward_propagate_model_weights(DMatrix[datum], params)
            cost = calculate_cost(Ŷ, Y[datum][1])
            acc = assess_accuracy(Ŷ, Y[datum][1])
            ∇ = back_propagate_model_weights(Ŷ, Y[datum][1], caches)
            params = update_model_weights(params, ∇, η)

            # update containers
            push!(costs, cost)
            push!(iters, i)
            push!(accuracy, acc)
            if findmax(Y[datum][1])[2] == findmax(Ŷ)[2][1]
                push!(yn, true)
            else
                push!(yn, false)
            end # if
            push!(YvŶ[1], mapping[findmax(Y[datum][1])[2]])
            push!(YvŶ[2], mapping[findmax(Ŷ)[2][1]])

            if verbose
                println("Datum: $datum, Cost: $cost, Accuracy: $acc")
            end # if

            # LOGGING
            if datum % 100 == 0 # only write every 100 characters
                open("case_names/logs/$d LOG.csv", "w") do io
                    CSV.write(io, DataFrame([iters, costs, accuracy, yn, YvŶ[1], YvŶ[2]], :auto), header=["Iteration", "Cost", "Accuracy", "Correct", "Answer", "Guess"])
                end # do
                open("case_names/logs/$d PAR.csv", "w") do io
                    CSV.write(io, DataFrame([collect(values(params))], :auto), header=collect(keys(params)))
                end # do
            end # if

        end # for

        if verbose
            cost = costs[end]
            acc = accuracy[end]
            println("Epoch -> $i, Most recent Cost -> $cost, Most recent Accuracy -> $acc")
        end # if

    end # for

    return (cost = costs, iterations = iters, accuracy = accuracy, parameters = params)

end # function