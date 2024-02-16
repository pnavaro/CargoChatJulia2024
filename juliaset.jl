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

function escapetime(z, c, lim, maxit)
    for n = 1:maxit
        abs(z) > lim && return n
        z = z*z + c
    end
    return maxit
end

function juliaset(x , y , c, lim , maxit )  
    nx = length(x)
    ny = length(y)
    julia = zeros(UInt8, (nx, ny))
    for i in eachindex(x), j in eachindex(y)
        z  = x[i] + 1im * y[j] 
        julia[j, i] = escapetime(z, c, lim, maxit)
    end
    return julia
end

# +
nx, ny = 512, 512
lim = 400
maxit = 100

x = LinRange(-1.6, 1.6, nx)
y = LinRange(-1.6, 1.6, ny)

c = -0.772691322542185 + 0.124281466072787im
juliaset(x , y , c, lim , maxit ) 

# +
using Colors, Images

const cmap = convert(Vector{RGB{N0f8}}, colormap("RdBu", maxit))
# -

typeof(cmap)

dump(cmap[end-3])

cmap[juliaset(x , y , c, lim , maxit )]

# +
using BenchmarkTools

@benchmark juliaset($x , $y , $c, $lim , $maxit )

# +
julia = zeros(UInt8, (nx, ny))
    
function juliaset!(julia, x , y , c, lim , maxit )
    for i in eachindex(x)
         for j in eachindex(y)
             z  = x[i] + 1im * y[j] 
             julia[j, i] = escapetime(z, c, lim, maxit)
        end
    end
end
# -

@benchmark juliaset!($julia, $x, $y, $c, $lim, $maxit)

nthreads()

cmap[julia]


