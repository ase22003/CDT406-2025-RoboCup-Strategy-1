using Flux

println("Hello, world!")

model=Chain(Dense(10, 5, relu), Dense(5, 2), softmax)

x=rand(Float32, 10)
y=model(x)

println("Output: ", y)