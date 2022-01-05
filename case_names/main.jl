include("train.jl")

using UnicodePlots

# get the data

# characters: [^0-9a-zA-Z&:,./()[]_-] THIS IS OLD
raw = []
open("case_names/data.txt") do f

    line = 0
    while !eof(f)
        push!(raw, readline(f))
    end # while

end # do

#println(raw[1:20])

#chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ -[]()0123456789"
chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ -"
raw = uppercase.(raw)
binary_split = [[[0 #==1/(length(chars)*2)==# for _ in 1:(length(chars)+1)] for i = 1:length(x)+1] for x in raw]
for s = 1:length(binary_split)

    for c = 1:length(raw[s])
        binary_split[s][c][findfirst(raw[s][c], chars)] = 1
    end # for
    binary_split[s][end][end] = 1

end # for

input_chars = 3
# split the cases to serve as inputs for the RNN.
DMatrix = Vector{Float64}[]
Y = Vector{Float64}[]

for s in binary_split
    
    for c = 1:length(s) -1

        push!(DMatrix, [])
        push!(Y, s[c+1])

        if c < input_chars
            for i = 1:(input_chars - c)
                append!( DMatrix[end], zeros(length(chars)) )
            end # for
        end # if
        if c != 0
            for i = (max(c - input_chars + 1, 1)):c
                append!( DMatrix[end], s[c][1:end-1] )
            end # for
        end # if

    end # for

end # for



input_dim = (length(chars)) * input_chars
hidden_dim = (length(chars)+1) * 2
output_dim = (length(chars)+1)
dims = [hidden_dim for i in 1:3]
prepend!(dims, input_dim)

append!(dims, output_dim)

println(string("Dimensions: ", dims))

nn_results = train_network(dims, reverse(DMatrix)[1:500], reverse(Y)[1:500], chars, epochs=50, η = 0.001)