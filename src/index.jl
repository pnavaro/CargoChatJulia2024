#ENV["GKSwstype"]="100" #src

#md # # CargoChat Julia 5 avril 2024
#md #
#md # ## Pierre Navaro
#md #
#md # Ingénieur Calcul à l'IRMAR (Institut de Recherche Mathématique de Rennes)
#md #
#md # Supports : https://pnavaro.github.io/CargoChatJulia2024
#md #
#md # Membre du [Groupe Calcul](https://calcul.math.cnrs.fr)
#md #
#md # Formation : Thèse de doctorat en Mécanique des fluides numerique
#md #
#md # Langages : Fortran, Python, Julia et un peu de R
#md #
#md #

#md # ---

#md # ![](assets/two-langage-1.png)
#
#md # ---

#md # ![](assets/two-langage-2.png)

#md # ---

#md #
#md # # Un peu de bibliographie
#md #
#md # L'article des créateurs:
#md #
#md # **Julia: A Fresh Approach to Numerical Computing**
#md # 
#md # *Jeff Bezanson, Alan Edelman, Stefan Karpinski, Viral B. Shah*  SIAM Rev., 59(1), 65–98. (34 pages) 2012
#md #
#md # Sources des images : [Matthijs Cox - My Target Audience](https://scientificcoder.com/my-target-audience#heading-the-two-culture-problem)
#md # 
#md # [Julia Hub](https://juliahub.com/) : le portail Julia pour éxécuter des notebooks et voir tous les packages existants.
#md # 
#md # Le livre blanc : [Why Julia ?](https://juliahub.com/assets/pdf/why_julia.pdf)
#md # 
#md # Un article en français : [A la découverte de Julia](https://zestedesavoir.com/articles/pdf/78/a-la-decouverte-de-julia.pdf)
#md # 
#md # Liste du CNRS avec la newsletter  https://github.com/pnavaro/NouvellesJulia
#md # 
#md # Liste rennaise pour les ateliers et les formations https://github.com/pnavaro/math-julia
#md # 


#md # ---

#md # # Une fonction Julia

f(x) = 2x^2

f(3)

#md # Les fonctions sont compilées lors du premier appel suivant le type de ses arguments

#md # ```julia
#md # @code_llvm f(3)
#md # 
#md # ;  @ In[22]:1 within `f`
#md # define i64 @julia_f_1553(i64 signext %0) #0 {
#md # top:
#md # ; ┌ @ intfuncs.jl:332 within `literal_pow`
#md # ; │┌ @ int.jl:88 within `*`
#md #     %1 = shl i64 %0, 1
#md # ; └└
#md # ; ┌ @ int.jl:88 within `*`
#md #    %2 = mul i64 %1, %0
#md # ; └
#md #   ret i64 %2
#md # }
#md # ```

#md # ---

#md # la fonction `f` peut accepter une matrice comme argument

A = rand(3,3)

f(A)

#md # la fonction `f` peut également opérer sur tous les éléments d'un tableau

b = [1, 2, 3]

map(f, b)

#md #

f.(b)


#md # ---

#md # # Algèbre linéaire

using LinearAlgebra

X = hcat(ones(3), rand(3), rand(3))
y = collect(1:3)

β = inv(X'X) * X'y

#md #

β = X \ y

#md # ---

#md # # Matrices creuses

using SparseArrays

rows = [1,3,4,2,1,3,1,4,1,5]
cols = [1,1,1,2,3,3,4,4,5,5]
vals = [5,-2,-4,5,-3,-1,-2,-10,7,9]

A = sparse(rows, cols, vals, 5, 5)

#md #

b = collect(1:5)

A \ b

#md # ---

#md # # Création d'un type Julia

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

#md #

Base.show(io::IO, r::MyRational) = print(io, "$(r.n) // $(r.d)")

b = MyRational(2, 3)

#md # ---

import Base.+
function +(a::MyRational, b::MyRational)
    MyRational(a.n*b.d+b.n*a.d, a.d*b.d)
end

a + b

#md # ---

#md # # Générer des données bruitées

using Random

function generate_data( rng, weights, bias; num_samples = 100, noise = 0.01)

    num_features = length(weights)
    x = randn(rng, (num_features, num_samples))  # création des variables explicatives aléatoires
    y =  x' * weights .+ bias .+ noise .* randn(rng, num_samples)

    return x, y

end

rng = Xoshiro(1234)

weights = [0.1, 0.2, 0.3, 0.4, 0.5 ]
bias = 0.2

x, y =  generate_data( rng, weights, bias)

#md # ---

#md # ## Fonction non typée pour une regression linéaire

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

#md # ---

linear_regression( x, y)

#md #

linear_regression( Float32.(x), Float32.(y))

#md # ---

#md # ## Fonction typée pour une regression linéaire

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

#md # ---

linear_regression( x, y)

#md #

linear_regression( Float32.(x), Float32.(y))

#md # --- 

#md # ## Type composé pour une régression linéaire

struct LinearRegression{T}

   learning_rate :: T
   iterations :: Int 
   weights :: Vector{T}
   bias :: T
    
   function LinearRegression( x :: Matrix{T}, y :: Vector{T}; learning_rate = 0.01, iterations = 1000) where T
        
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
        
        new{T}( learning_rate, iterations, weights, bias)
        
    end
        
end
    

#md # --- 

model = LinearRegression(x, y)

model.weights

#md #

model.bias

#md #

predict(model, x) = x' * model.weights .+ model.bias

function Base.show(io :: IO, model :: LinearRegression) 
    println(io, "Linear Regression")
    println(io, "=================")
    println(io, "weights : $(round.(model.weights, digits=3))")
    println(io, "bias : $(round(model.bias, digits=3))")
end

model

#md #

#md # ---

#md # # Les particularités de Julia
#md #
#md # - Les types composés ne pourront pas être modifiés dans une même session, peu pratique mais offre une meilleure gestion de la mémoire.
#md # - Toutes les fonctions sont compilés lors de leur premier appel, un peu de lattence mais l'exécution est beaucoup plus rapide ensuite.
#md # - Les fonctions sont spécialisées suivant le type de leurs arguments, plus de compilation mais cela améliore la lisibilité de vos interfaces.
#md # - Julia est interactif, facile à utiliser mais la création de binaires est compliquée.
#md # - Julia possède plusieurs manières pour manipuler les tableaux, très pratique pour optimiser son empreinte mémoire mais nécessite un apprentissage.
#md # - La bibliothèque standard et quasiment tous les packages sont écrits en Julia, il est facile de s'en inspirer.
#md # - Il y a un mécanisme de géneration de code et de méta-programmation qui permet de faire de très belles interfaces, il nécessite des compétences avancées.
#md #

#md # ---
