#ENV["GKSwstype"]="100" #src

#md # # CargoChat Julia 5 avril 2024
#md #
#md #  - *Pierre Navaro*
#md #
#md #  - Ingénieur Calcul à l'IRMAR
#md #
#md #  https://pnavaro.github.io/CargoChatJulia2024
#md #

#md # ---

#md # ![](assets/two-langage-1.png)
#
#md # ---

#md # ![](assets/two-langage-2.png)

#md # ---

#md #
#md # # Pourquoi Julia?
#md #
#md # - Démarré en 2009,  première version en 2012 et version stable depuis 2018.
#md # - Les langages comme Python et R sont faciles à prendre en main mais sont parfois lents.
#md # - Les langages comme Fortran et C/C++ sont plus difficiles à utiliser mais sont rapides.
#md # - Julia propose d'éliminer "le problème des deux langages" pour offrir les meilleur des deux mondes.
#md #
#md # **Julia: A Fresh Approach to Numerical Computing**
#md # 
#md # *Jeff Bezanson, Alan Edelman, Stefan Karpinski, Viral B. Shah*
#md # 
#md # SIAM Rev., 59(1), 65–98. (34 pages) 2012


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


#md # # Générer des données bruitées

using Random

rng = Xoshiro(1234)

num_samples, num_features = 100, 2

x = randn(rng, (num_features, num_samples))

weights = [0.5, 0.1]

bias = 0.2

noise = 0.01 .* randn(rng, num_samples)

y =  x' * weights .+ bias .+ noise

#md # ---

#md # ## Fonction pour une regression linéaire

function linear_regression( x, y; learning_rate = 0.01, iterations = 1000)
        
    T = eltype(x)
    @assert eltype(y) <: T
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

#md # 

linear_regression( Float16.(x), Float16.(y))

#md # ---

#md # ## Type composé pour une régression linéaire

struct LinearRegression{T}

   learning_rate :: T
   iterations :: Int 
   weights :: Vector{T}
   bias :: T
    
   function LinearRegression( x :: Matrix{T}, y; learning_rate = 0.01, iterations = 1000) where T
        
        num_features, num_samples  = size(x)
        weights = ones(T, num_features)
        bias = zero(T)
        
        for i in 1:iterations
            
            y_pred =  x' * weights .+ bias
            dw = x * ( y_pred .- y ) ./ num_samples
            db = sum(y_pred .- y ) ./ num_samples
            weights .-= learning_rate .* dw
            bias -= learning_rate * db
            
        end
        
        new{T}( learning_rate, iterations, weights, bias)
        
    end
        
end
    


model = LinearRegression(x, y)

model.weights

model.bias

predict(model, x) = x' * model.weights .+ model.bias

function Base.show(io :: IO, model :: LinearRegression) 
    println("Linear Regression")
    println("weights : $(model.weights)")
    println("bias : $(model.bias)")
end
