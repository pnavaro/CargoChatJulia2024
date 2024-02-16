# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Julia 1.10.0
#     language: julia
#     name: julia-1.10
# ---

f(x) = x^3 - x

f(3)

@code_llvm debuginfo = :none f(3)

@code_llvm debuginfo = :none f(1.5)

# +
using Plots

plot(f, xlims=(-2, 2), ylims = (-2, 2), framestyle = :origin, aspect_ratio = 1, label = "y = x³ - x")
# -

A = rand(5,5)

f(A)

b = [-2, -1, 0, 1, 2]

f(b)

map(f, b)

f.(b)

# +
using LinearAlgebra

A \ b
# -

Q, R = qr(A)
(inv(factorize(R))Q') * b

U, S, V = svd(A)
V * Diagonal(1 ./ S) * U' * b

# +
using SparseArrays

rows = [1,3,4,2,1,3,1,4,1,5]
cols = [1,1,1,2,3,3,4,4,5,5]
vals = [1,-2,-4,5,-3,-1,-2,1,7,-1]

A = sparse(rows, cols, vals, 5, 5)

# +
b = collect(1:5)

A * b
# -

vcat(ones(5)', rand(4,5))

# +
using Random

function generate_data( rng, β; num_samples = 100, noise = 0.01)

    num_features = length(β) - 1
    x = vcat(ones(num_samples)', randn(rng, (num_features, num_samples)))  # création des variables explicatives
    y =  x' * β .+ noise .* randn(rng, num_samples)

    return x, y

end

rng = Xoshiro(1234)

weights = [0.1, 0.2, 0.3, 0.4, 0.5 ]
bias = 1.0
β = [bias, weights...]
# -

x, y =  generate_data( rng, β)

# +

function linear_regression( x, y, β ; learning_rate = 0.01, iterations = 1000)
        
    num_features, num_samples  = size(x)
    
    for i in 1:iterations
        
        y_pred =  x' * β
        dβ = x * ( y_pred .- y ) ./ num_samples
        β .-= learning_rate .* dβ
        
    end
    
    return β     
end

β = ones(size(x,1))

w  = linear_regression( x, y, β)
# -

β = ones(Float32, size(x,1))
linear_regression( Float32.(x), Float32.(y), β)

# +
using CUDA

β = CUDA.ones(size(x,1))
linear_regression(cu(x), cu(y), β)
# -

weights = rand(100)
bias = 1.0
β = [bias, weights...]
x, y =  generate_data( rng, β, num_samples=100_000)

@time linear_regression( x, y, β)

β = CUDA.ones(size(x,1))
CUDA.@time linear_regression(cu(x), cu(y), β)

# +
struct LinearRegression{T}

   learning_rate :: T
   iterations :: Int 
   β :: Vector{T}
    
   function LinearRegression( x :: Matrix{T}, y :: Vector{T}; learning_rate = 0.01, iterations = 1000) where T
        
        num_features, num_samples  = size(x)
        β = Vector{T}(undef, num_features)
        linear_regression(x, y, β; learning_rate = learning_rate, iterations = iterations)
        
        new{T}( learning_rate, iterations, β)
        
    end
        
end


model = LinearRegression(x, y)

model.β

# +
predict(model :: LinearRegression, x) = x' * model.β

function Base.show(io :: IO, model :: LinearRegression) 
    println(io, "Linear Regression")
    println(io, "=================")
    println(io, "β : $(round.(model.β, digits=3))")
end

model
# -

model_f0 = LinearRegression(Float32.(x), Float32.(y))

model_f0.β

model_gpu = LinearRegression(cu(x), cu(y))




