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

det(A)

# +
using SparseArrays

rows = [1,3,4,2,1,3,1,4,1,5]
cols = [1,1,1,2,3,3,4,4,5,5]
vals = [1,-2,-4,5,-3,-1,-2,1,7,-1]

A = sparse(rows, cols, vals, 5, 5)

# +
b = collect(1:5)

A * b

# +
"""
Nombre rationnel
"""
struct MyRational
    n :: Int
    d :: Int

    function MyRational(n :: Int, d :: Int) 
        @assert d != "zero denominator"
        g = gcd(n,d)
        new( n ÷ g, d ÷ g)
    end
end

a = MyRational(3, 4)

# +
Base.show(io::IO, r::MyRational) = print(io, "$(r.n) / $(r.d)")

b = MyRational(2, 3)

# +
import Base.+
function +(a::MyRational, b::MyRational)
    MyRational(a.n*b.d+b.n*a.d, a.d*b.d)
end

a + b

# +
using Random

function generate_data( rng, weights, bias; num_samples = 100, noise = 0.01)

    num_features = length(weights)
    x = randn(rng, (num_features, num_samples))  # création des variables explicatives aléatoires
    y =  x' * weights .+ bias .+ noise .* randn(rng, num_samples)

    return x, y

end

rng = Xoshiro(1234)

weights = [0.1, 0.2, 0.3, 0.4, 0.5 ]
bias = 1.0

x, y =  generate_data( rng, weights, bias)

# +
function linear_regression( x, y; learning_rate = 0.01, iterations = 1000)
        
    num_features, num_samples  = size(x)
    weights = ones(num_features)
    bias = 0.0
    
    for i in 1:iterations
        
        y_pred =  x' * weights .+ bias
        dw = x * ( y_pred .- y ) ./ num_samples
        db = sum(y_pred .- y ) ./ num_samples
        weights .-= learning_rate .* dw
        bias -= learning_rate * db
        
    end
    
    return weights, bias
        
end

w , b = linear_regression( x, y)
# -

linear_regression( Float32.(x), Float32.(y))

# +
function linear_regression( x::Matrix{T}, y::Vector{T}; learning_rate = 0.01, iterations = 1000) where T
        
    num_features, num_samples  = size(x)
    weights = ones(T, num_features)
    bias = zero(T)
    
    for i in 1:iterations
        
        y_pred =  x' * weights .+ bias
        dw = x * ( y_pred .- y ) ./ num_samples
        db = sum(y_pred .- y ) ./ num_samples
        weights .-= T(learning_rate) .* dw
        bias -= T(learning_rate) * db
        
    end
    
    return weights, bias
        
end

linear_regression( Float32.(x), Float32.(y))

# +
struct LinearRegression{T}

   learning_rate :: T
   iterations :: Int 
   weights :: Vector{T}
   bias :: T
    
   function LinearRegression( x :: Matrix{T}, y :: Vector{T}; learning_rate = 0.01, iterations = 1000) where T
        
        num_features, num_samples  = size(x)
        weights, bias = linear_regression(x, y, learning_rate = learning_rate, iterations = iterations)
        
        new{T}( learning_rate, iterations, weights, bias)
        
    end
        
end


model = LinearRegression(x, y)

model.weights
# -

model.bias

# +
predict(model :: LinearRegression, x) = x' * model.weights .+ model.bias

function Base.show(io :: IO, model :: LinearRegression) 
    println(io, "Linear Regression")
    println(io, "=================")
    println(io, "weights : $(round.(model.weights, digits=3))")
    println(io, "bias : $(round(model.bias, digits=3))")
end

model
# -

model_f0 = LinearRegression(Float32.(x), Float32.(y))

model_f0.bias


