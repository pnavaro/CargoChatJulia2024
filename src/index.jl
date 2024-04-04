#ENV["GKSwstype"]="100" #src

#md # # CargoChat Julia 5 avril 2024
#md #
#md # https://cargo.resinfo.org
#md #
#md # ## Pierre Navaro
#md #
#md # Ingénieur Calcul à l'IRMAR (Institut de Recherche Mathématique de Rennes)
#md #
#md # Supports : https://pnavaro.github.io/CargoChatJulia2024
#md #
#md # Membre du [Groupe Calcul](https://calcul.math.cnrs.fr)
#md #
#md # Formation : Thèse de doctorat en Mécanique des fluides.
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




#md # # Une fonction Julia

f(x) = 2x^2

#md # Les fonctions sont compilées lors du premier appel suivant le type des arguments

#md # ```julia
#md # @code_llvm debuginfo = :none f(3)
#md # ```
import InteractiveUtils #hide
InteractiveUtils.code_llvm(f, (Int,); debuginfo = :none) #hide
#md # ```julia
#md # @code_llvm debuginfo = :none f(1.5)
#md # ```
import InteractiveUtils #hide
InteractiveUtils.code_llvm(f, (Float64,); debuginfo = :none) #hide

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

X = hcat(ones(3), rand(3), rand(3)) # initialise une matrice avec 3 colonnes
y = collect(1:3) # collect transforme l'itérateur en tableau

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

A \ b # le solveur utilisé dépendra du type de la matrice, creuse, symétrique, définie positive...

#md # ---

#md # # Création d'un type Julia

"""
    MyRational(n, d)

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

a = MyRational(14, 8)

#md #

Base.show(io::IO, r::MyRational) = print(io, "$(r.n) // $(r.d)")

b = MyRational(14, 8)

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

#md # ## Fonction non typée pour une régression linéaire

function linear_regression( x, y; learning_rate = 0.01, iterations = 1000)
        
    num_features, num_samples  = size(x)
    weights = ones(num_features)
    bias = 0.0
    
    for i in 1:iterations
        
        y_pred =  x' * weights .+ bias
        dw = x * ( y_pred .- y ) ./ num_samples
        db = sum(y_pred .- y ) / num_samples
        weights .-= learning_rate .* dw
        bias -= learning_rate * db
        
    end
    
    return weights, bias
        
end

#md # ---

linear_regression( x, y)

#md #

linear_regression( Float32.(x), Float32.(y))

#md # Avec des flottants simple précision en entrée, on aimerait que tous les calculs soient faits en simple précision

#md # Si le code source de la méthode est bien écrit, le code source et le type concret de tous les arguments sont des informations suffisantes pour que le compilateur puisse déduire le type concret de chaque variable au sein de la fonction. La fonction est alors dite "type stable" et le compilateur Julia produira un code efficace.

#md # Si, pour diverses raisons, le type d'une variable locale ne peut pas être déduit des types des arguments, le compilateur produit, un code machine beaucoup de structures conditionnelles, couvrant toutes les options de ce que le type de chaque variable pourrait être. La perte de performance est souvent significative, facilement d'un facteur 10.

#md # **Évitez les méthodes qui renvoient des variables dont le type dépend de la valeur.**
#md #
#md # **Dans la mesure du possible, utilisez des tableaux avec un type d'élément concret.**
#md #
#md # ---

#md # ## Fonction typée pour une régression linéaire

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

#md # # Les particularités du langage Julia
#md #
#md # - Les types composés ne pourront pas être modifiés dans une même session, peu pratique mais offre une meilleure gestion de la mémoire.
#md # - Toutes les fonctions sont compilées lors de leur premier appel, un peu de latence, mais l'exécution est beaucoup plus rapide ensuite.
#md # - Les fonctions sont spécialisées suivant le type de leurs arguments, plus de compilation, mais cela améliore la lisibilité de vos interfaces.
#md # - Julia est interactif, facile à utiliser, mais la création de binaire est compliquée.
#md # - Julia possède plusieurs manières pour manipuler les tableaux, très pratique pour optimiser son empreinte mémoire, mais nécessite un apprentissage.
#md # - La bibliothèque standard et quasiment tous les packages sont écrits en Julia, il est facile de s'en inspirer.
#md # - Il y a un mécanisme de génération de code et de métaprogrammation qui permet de faire de très belles interfaces. Il nécessite des compétences avancées et le déverminage est plus difficile.
#md # - L'interface avec R et Python est très facile, le C et le Fortran sont également nativement encapsulables. En revanche, le C++ c'est plus compliqué...

#md # ---
#md #
#md # # Ce qu'il faut savoir avant de se lancer...

#md # - L'apprentissage profond en Julia est en retard par rapport à ce que proposent les bibliothèques proposées en Python. Il y a tout de même des choses intéressantes en différentiation automatique.
#md # - Beaucoup de packages disponibles sont non maintenus ou complètement abandonnés. Il faut être prudent lorsque l'on choisit ses dépendances. Si la licence le permet, on peut toujours récupérer des choses dans les packages, ils sont écrits en Julia.
#md # - Le "workflow" à mi-chemin entre le langage interprété et le langage compilé est parfois déroutant et peut dégoûter les débutants. Les durées de compilations ont été fortement réduites depuis la version 1.
#md # - L'IDE préconisé est VSCode, mais il a moins de fonctionnalités que ce qui est proposé dans les autres langages.
#md # - L'optimisation de performance en Julia nécessite un apprentissage particulier. On peut être déçu après une traduction naïve depuis du Fortran ou du MATLAB. C'est aussi la partie la plus "satisfaisante" du langage et attention cela peut devenir addictif :-)
#md # - Les opérations vectorisées sont moins efficaces en Julia et cela peut être décevant si l'on est attaché à cette manière de coder les algorithmes. En revanche, vous pouvez faire autant de boucles que vous voulez !
#md # - En Julia, il faut faire des fonctions pour minimiser les calculs et les allocations dans l'environnement global. Les erreurs dues à la portée des variables sont parfois un peu déroutantes.

#md # ---
#md #
#md # # Pourquoi il faut se lancer !
#md #
#md # - **C'est beau** : une syntaxe lisible très proche des mathématiques 
#md # - **C'est puissant** : un langage bien conçu, le "multiple dispatch" et la possibilité de l'optimiser de manière incrémentale en ajoutant progressivement les types, par exemple, est agréable. Il y a beaucoup de fonctions haut-niveau disponibles qui permettent d'écrire des algorithmes complexes en quelques lignes de codes.
#md # - **C'est fait pour nous** : c'est un langage fait pour les sciences qui est gouverné par des scientifiques. L'écosystème est déjà très large dans beaucoup de disciplines. Pour les équations différentielles et l'optimisation mathématique, Julia est déjà populaire.
#md # - **C'est facile** : l'apprentissage est rapide et l'accès au calcul sur GPU, par exemple, est très simple.
#md # - **C'est interactif** : Le Julia REPL est puissant et très pratique et on peut utiliser les notebooks Jupyter ou Pluto.
#md # - **C'est rapide** : Le code est compilé, la parallélisation est proposée nativement et très accessible. C'est un candidat tout à fait crédible pour un projet HPC.
#md # - **C'est ouvert** : Le langage Julia est publié sous une licence MIT. Le système de gestion des packages est très efficace et c'est très simple de créer et proposer son propre package.
#md #
#md # ---

#md #
#md # #  Bibliographie
#md #
#md # L'article des créateurs:
#md #
#md # **Julia: A Fresh Approach to Numerical Computing**
#md # 
#md # *Jeff Bezanson, Alan Edelman, Stefan Karpinski, Viral B. Shah*  SIAM Rev., 59(1), 65–98. (34 pages) 2012
#md #
#md # Sources des images : [Matthijs Cox - My Target Audience](https://scientificcoder.com/my-target-audience#heading-the-two-culture-problem)
#md # 
#md # [Julia Hub](https://juliahub.com/) : le portail Julia pour exécuter des notebooks et voir tous les packages existants.
#md # 
#md # Le livre blanc : [Why Julia ?](https://juliahub.com/assets/pdf/why_julia.pdf)
#md # 
#md # Un article en français : [A la découverte de Julia](https://zestedesavoir.com/articles/pdf/78/a-la-decouverte-de-julia.pdf)
#md #
#md # Un support pour se lancer : [Modern Julia Workflows](https://modernjuliaworkflows.github.io)
#md # 
#md # Liste du CNRS avec la newsletter  https://github.com/pnavaro/NouvellesJulia
#md # 
#md # Liste rennaise pour les ateliers et les formations https://github.com/pnavaro/math-julia
#md # 
