using Flux
using Flux: onehotbatch, onecold
using Optimisers
using Random
using Statistics

#example code for classifying 2d points into two classes, needs julia and some packages installed to run

function main()
# 1. Generate synthetic data
Random.seed!(1234)
n_samples = 100
X1 = randn(2, n_samples) .+ [-2.0; -2.0] # Class 0
X2 = randn(2, n_samples) .+ [2.0; 2.0] # Class 1

X = hcat(X1, X2) # Input: 2 × 200
y = vcat(fill(0, n_samples), fill(1, n_samples)) # Labels: 0s and 1s
Y = Flux.onehotbatch(y, 0:1) # One-hot encoded: 2 × 200

# 2. Define model
model = Chain(
Dense(2, 8, relu),
Dense(8, 2),
softmax
)

# 3. Define loss function
function loss_fn(model)
ŷ = model(X)
return Flux.crossentropy(ŷ, Y)
end

# 4. Optimizer setup
opt = Descent(0.01)
opt_state = Optimisers.setup(opt, model)

# 5. Training loop
for epoch in 1:100
loss, grads = Flux.withgradient(loss_fn, model)
opt_state, model = Optimisers.update(opt_state, model, grads)

if epoch % 10 == 0
println("Epoch $epoch: loss = $(round(loss, digits=4))")
end
end

# 6. Evaluate
preds = model(X)
predicted_labels = onecold(preds, 0:1)
accuracy = mean(predicted_labels .== y)
println("Training accuracy: $(round(accuracy * 100, digits=2))%")
end

# Run it
main()