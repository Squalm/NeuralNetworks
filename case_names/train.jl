using DataFrames, CSV, Dates, ProgressBars

include("initialise.jl")
include("forward_propagation.jl")
include("cost.jl")
include("check_accuracy.jl")
include("backpropagation.jl")
include("gradient_descent.jl")

"""
Train the network using the desired architecture by matching the training inputs (DMatrix) to their
corresponding outputs (Y) over some number of iterations (epochs) and a learning rate (η).
"""
function train_network(layer_dims::Vector{Int64}, DMatrix::Vector{Vector{Float64}}, Y::Vector{Vector{Float64}}, mapping::String; η = 0.001, epochs = 500, seed = 758907)

    η_first = η
    costs = BigFloat[BigFloat(0.0) for x in 1:length(DMatrix) * epochs]
    iters = Int[trunc((x-1) / length(DMatrix))+1 for x in 1:length(DMatrix) * epochs]
    confidence = BigFloat[BigFloat(0.0) for x in 1:length(DMatrix) * epochs]
    YvŶ = Vector{Char}[Char[' ' for x in 1:length(DMatrix) * epochs], Char[' ' for x in 1:length(DMatrix) * epochs]]
    mapping *= "#"

    # initialise
    params = initialise_model(layer_dims, seed)

    # LOGGING
    d = string(Dates.now())[1:13] * "-" * string(Dates.now())[15:16]

    # train
    for i = 1:epochs

        Ŷ = Any[[] for x in 1:length(DMatrix)]
        caches = [[] for x in 1:length(DMatrix)]

        iter = ProgressBar(1:length(DMatrix))
        set_description(iter, "Epoch $i 1/6 Forward Propagating")

        Threads.@threads for datum in iter
            Ŷ[datum], caches[datum] = forward_propagate_model_weights(DMatrix[datum], params)
        end # for

        set_description(iter, "Epoch $i 2/6 Calculating Costs") # I sorta modified this step...

        Threads.@threads for datum in iter
            costs[(i-1) * length(DMatrix) + datum] = calculate_cost(Ŷ[datum], Y[datum])
        end # for

        set_description(iter, "Epoch $i 3/6 Calculating Confidence")

        Threads.@threads for datum in iter
            confidence[(i-1) * length(DMatrix) + datum] = assess_accuracy(Ŷ[datum], Y[datum])
        end # for

        set_description(iter, "Epoch $i 4/6 Calculating Gradients")

        ∇ = Any[missing for x in 1:length(DMatrix)]
        Threads.@threads for datum in iter
            ∇[datum] = back_propagate_model_weights(Ŷ[datum], Y[datum], caches[datum])
        end # for

        iter = ProgressBar(1:length(collect(keys(params))))
        set_description(iter, "Epoch $i 5/6 Updating Model Params")
        ∇_avg = Dict()

        for x in iter
            ∇_avg[string("δ", collect(keys(params))[x])] = sum([δ[string("δ", collect(keys(params))[x])] for δ in ∇]) / length(∇)
        end # for

        params = update_model_weights(params, ∇_avg, η)

        iter = ProgressBar(1:length(DMatrix))
        set_description(iter, "Epoch $i 6/6 Logging calcs")

        for datum in iter
            YvŶ[1][(i-1) * length(DMatrix) + datum] = mapping[findmax(Y[datum])[2]]
            YvŶ[2][(i-1) * length(DMatrix) + datum] = mapping[findmax(Ŷ[datum])[2][1]]
        end # for

        open("case_names/logs/$d LOG.csv", "w") do io
            CSV.write(io, DataFrame([iters, costs, confidence, [(YvŶ[1][x] == YvŶ[2][x]) for x in 1:length(YvŶ[1])], YvŶ[1], YvŶ[2]], :auto), header=["Iteration", "Cost", "Accuracy", "Correct", "Answer", "Guess"])
        end # do
        open("case_names/logs/$d PAR.csv", "w") do io
            CSV.write(io, DataFrame([collect(values(params))], :auto), header=collect(keys(params)))
        end # do

    end # for

    return (cost = costs, iterations = iters, confidence = confidence, parameters = params)

end # function