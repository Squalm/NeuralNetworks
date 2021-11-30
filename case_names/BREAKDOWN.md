# Breakdown

This file is intended to explain what's going on in this NN with relatively broad strokes so it's understandable to someone with little to no experience in NNs.

N.B. Please open in something that supports LaTeX maths-y stuff. Basically any half decent markdown previewer will do this.

## `main.jl`

Here process the data in `data.txt` to be inputs into our neural net. It takes each line and splits it into 15 character long chunks (with overlap, and allowing blank characters at the beginning).

Take for example:

```
AG v Wilkinson -- Ahmad Ors
```

First we make each character into the string into an array with one 1 showing the character so `A` would generate an array like

```julia
Int[..., 0, 0, 1, 0, 0, ...]
```

Each value in the array is a specific character, with a special character at the end for stop. 

Lets assume our input length is 5 characters, this is like the NNs memory. To create the first datum for our NN we add 4 (input length -1) blank arrays and the array we just generated (flattening all of these as we go). For the next character, we add 3 (input length -2) blank sets `A` and then the array for `G`. Once we reach 5 characters into the string we remove characters so that it only sees the last 5. For example datum 6 would be `"G v W"`. 

This gives us a lot of data points which can use as training data. So we can pass that (as `DMatrix`) but we need some other parameters first:

```julia
function train_network(layer_dims::Vector{Int64}, DMatrix::Vector{Any}, Y::Vector{Any}; η = 0.001, epochs = 500, seed = 1337, verbose = true)
```

Layer dims looks something like this:

```julia
Int64[345, 207, 207, 207, 69]
```

Where each value describes the number of neurons in a layer of the NN (the first and last layers describe the input and output layers respectively). `layer_dims` is a hyperparameter, which means that if we wanted to optimise the NN (and had lots of resources) we could train the NN using some kind of algorithm that worked out how many layers we need (we want to minimise layers where we can, to avoid doing extra calculations, but we don't want to have so few our model can't be accurate). The lengths of the layers are arbitrary but the input layer must match (in our example it's $5 * 69 = 345$) the length of the arrays in `DMatrix` and the output layer must match the length of the arrays in `Y`. 

`Y` is the set of the actual values from our dataset for each item in `DMatrix`. We have been calculating these alongside our inputs so that the two vectors line up. It's the next character in our string (because that's the one we want the NN to guess). 

## `train.jl`

This contains the function `train_network` which I briefly talked about above which outlines the by which process our NN is going to take to learn. `η` describes the learning (more on this later!). `epochs` tells our code how many times to go through the whole training dataset. `seed` is used instead of the default RNG so we get consistent results (useful to recreate situations). `verbose` tells our code whether it should print to console as it does things.  
First we have to initialise our model, and for this we use `initialise_model(layer_dims, seed)` which can be found in...

## `initialise.jl`

```julia
function initialise_model(layer_dims, seed)
```
This function creates and returns a dictionary `params` which contains all the initial weights and biases for the model. The biases start at 0 but the weights are randomised to avoid symmetry (we want different parts of our NN to learn to do different things but if the weights all start the same, they would never be different so there might as well only be one neuron in each layer).  

The next thing to do in `train.jl` is to start running our NN! First we propagate forward through the model in...

## `forward_propagation.jl`

`forward_propagate_model_weights(DMatrix, params)` loops through each layer of the NN and gets the values:

For layer $l$

$$ Z_{l,n} = W_{l,n} * A_{l-1} + b $$

So for each neuron in each layer (except the input layer) we multiply each value in the previous layer by some number (specific to each neuron), and then add a bias (find this in `linear_steps.jl`). We're not quite done there for each layer. We actually need to apply an activation function so our data is remotely usable. Most of the time we only need to use a reLU function which simply returns a 0 if the value is negative and otherwise returns the value. However, in the final layer we use a sigmoid activation function to obtain a value always between 0 and 1.

$$ A_{l,n} = \frac 1 {1 + e^{-Z_{l,n}}} $$

The greater Z is the closer this will be to 1, and the smaller Z the closer to 0. Usefully it can never fall outside of those bounds (find activation functions in `activations_functions.jl`).

Now that we've iterated through our NN, we need to calculate the cost of our results in...

## `cost.jl`

For now we are using an implementation of binary cross entropy which says that if you have two functions $p(x)$ and $q(x)$ you can calculate the distance between the results of the two functions essentially like this:

$$ C = q(x) * \log_2 (\frac 1 {p(x)}) $$

The derivation for this is quite complicated but a very readable explanation can be found here: [Visual Information](https://colah.github.io/posts/2015-09-Visual-Information/). I don't actually know whether or not this the correct cost function for this NN, but it should be close enough that it doesn't matter.  


Our implementation of this looks slightly different:

$$ C = \frac {- \sum ( Y * \log_2 \hat Y + (1 - Y) * \log_2 (1 - \hat Y)) } m $$

Where $m$ is the size of $Y$. This actually gives us the same thing because or actual values ($Y$) are binary. When the actual value is 0 that element of the sum can be simplified to:

$$ 1 * \log_2 (1 - \hat Y) $$

This is almost the same form as above. We've got $1 - \hat Y$ rather than $\frac 1 {1 - \hat Y}$. I've accounted for this by taking that out as a minus (simple log laws). This affects every term so I took it out of the sum. A similar thing is obtained if $Y$ is 1:

$$ 1 * \log_2 \hat Y $$

Next we need to do the really interesting bit: actually learning in...

## `backpropagation.jl`

We can think about this quite clearly if we imagine a neural network with only one neuron per layer. Essentially we need to take the value we got, and modify our NN to produce a value that's slightly closer to our desired value. Lets take a neuron, $n$, it has a weight, $w$, and bias, $b$. It received an input ($A$) 0, and we want it to output 1. We can the output we got, $Y_n$, and the output we wanted $\hat Y_n$.

$$ \hat Y_n = wA + b $$

It's easy to see how we could optimise one neuron to create the result we want. Substitute for $A$ and our desired $Y_n$. e.g.

$$ 1 = 0w + b $$

Picking random values for w and letting b equal 0 initially (as we do in our NN) currently gives us:

$$ 0*0.74... + 0 = 0 $$

We can clearly see that $w$ does not affect our result in this case, so we can modify $b$ only:

$$ b = b + (Y_n - b) \eta $$

Where $\eta$ is the learning rate.

$$ b = 0 + (1 - 0) * 0.01 = 0.01 $$

So we've modified our value of $b$ to give a slightly better answer than before. We could let $\eta = 1$ then we would correctly guess the right answer for this situation! However, this is called over-optimisation and doesn't actually solve the problem we want it to. Our NN would always be completely overhauled each time we backpropagate to get the ideal result but we would never converge on something which can produce and accurate result from data it hasn't seen before (or even from other data in our dataset, if $\eta = 1$). Even in less extreme cases, over-optimisation is still a big issue. If your NN learns for too long it may learn to fit perfectly to your data and unlearn how to solve the problem which means it would produce less accurate results when it sees data it's never seen before! (This is also why it's important to keep some data for unsupervised tests after supervised training.)

The partial derivatives of each of the variables with respect to $Y$ can be calculated as follows:

$$ \frac {\delta \hat Y} {\delta Y} = - \frac Y {\hat Y} + \frac {1 - Y}{1 - \hat Y} $$

***WORK IN PROGRESS**

## Summary

Our neural network follows a simple loop: take some values and predict some values then get the cost of our predictions and compute the gradients of the parameters with respect to the cost function and update our parameters using gradients optimisation.  

At some point we hope the NN will be good enough at guessing values we can give it any prompt that it's never seen before and it could fill in a reasonable sounding court case title for that.

Huge thanks to [this tutorial](https://towardsdatascience.com/how-to-build-an-artificial-neural-network-from-scratch-in-julia-c839219b3ef8) from which I copied a lot of the code and structure. The tutorial's code required a lot of bug fixing (and functionality fixing) to run in newer version of julia (1.6) so if you're looking to do this yourself, I might recommend you use my code rather than Bernard Brenyah's.