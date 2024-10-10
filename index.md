






# CargoChat Julia 5 avril 2024


https://cargo.resinfo.org






## Pierre Navaro


Ingénieur Calcul à l'IRMAR (Institut de Recherche Mathématique de Rennes)


Supports : https://pnavaro.github.io/CargoChatJulia2024


Membre du [Groupe Calcul](https://calcul.math.cnrs.fr)


Formation : Thèse de doctorat en Mécanique des fluides.


Langages : Fortran, Python, Julia et un peu de R


---


![](assets/two-langage-1.png)


---


![](assets/two-langage-2.png)


---






# Une fonction Julia


```julia
f(x) = 2x^2
```


```
f (generic function with 1 method)
```


Les fonctions sont compilées lors du premier appel suivant le type des arguments


```julia
@code_llvm debuginfo = :none f(3)
```


```
; Function Signature: f(Int64)
define i64 @julia_f_6645(i64 signext %"x::Int64") #0 {
top:
  %0 = shl i64 %"x::Int64", 1
  %1 = mul i64 %0, %"x::Int64"
  ret i64 %1
}
```


```julia
@code_llvm debuginfo = :none f(1.5)
```


```
; Function Signature: f(Float64)
define double @julia_f_6665(double %"x::Float64") #0 {
top:
  %0 = fmul double %"x::Float64", %"x::Float64"
  %1 = fmul double %0, 2.000000e+00
  ret double %1
}
```


---


la fonction `f` peut accepter une matrice comme argument


```julia
A = rand(3,3)

f(A)
```


```
3×3 Matrix{Float64}:
 1.50854   2.113     0.838307
 0.16218   0.258103  0.192875
 0.302459  0.395058  0.572488
```


la fonction `f` peut également opérer sur tous les éléments d'un tableau


```julia
b = [1, 2, 3]

map(f, b)
```


```
3-element Vector{Int64}:
  2
  8
 18
```


```julia
f.(b)
```


```
3-element Vector{Int64}:
  2
  8
 18
```


---






# Algèbre linéaire


```julia
using LinearAlgebra

X = hcat(ones(3), rand(3), rand(3)) # initialise une matrice avec 3 colonnes
y = collect(1:3) # collect transforme l'itérateur en tableau

β = inv(X'X) * X'y
```


```
3-element Vector{Float64}:
  6.3463624262875555
 -4.00064980680159
 -6.291102866053052
```


```julia
β = X \ y
```


```
3-element Vector{Float64}:
  6.346362426287825
 -4.0006498068019525
 -6.291102866053287
```


---






# Matrices creuses


```julia
using SparseArrays

rows = [1,3,4,2,1,3,1,4,1,5]
cols = [1,1,1,2,3,3,4,4,5,5]
vals = [5,-2,-4,5,-3,-1,-2,-10,7,9]

A = sparse(rows, cols, vals, 5, 5)
```


```
5×5 SparseArrays.SparseMatrixCSC{Int64, Int64} with 10 stored entries:
  5  ⋅  -3   -2  7
  ⋅  5   ⋅    ⋅  ⋅
 -2  ⋅  -1    ⋅  ⋅
 -4  ⋅   ⋅  -10  ⋅
  ⋅  ⋅   ⋅    ⋅  9
```


```julia
b = collect(1:5)

A \ b # le solveur utilisé dépendra du type de la matrice, creuse, symétrique, définie positive...
```


```
5-element Vector{Float64}:
 -1.0753295668549907
  0.4
 -0.8493408662900187
  0.03013182674199632
  0.5555555555555556
```


---






# Création d'un type Julia


```julia
"""
    MyRational(n, d)

Nombre rationnel
"""
struct MyRational

    n :: Int
    d :: Int

    function MyRational(n :: Int, d :: Int)
        @assert  d != 0 "dénominateur nul"
        g = gcd(n,d)
        new( n ÷ g, d ÷ g)
    end

end
```


```
Main.var"ex-index".MyRational
```


---


```julia
a = MyRational(14, 8)
```


```
Main.var"ex-index".MyRational(7, 4)
```


```julia
Base.show(io::IO, r::MyRational) = print(io, "$(r.n) // $(r.d)")

b = MyRational(5, 9)
```


```
5 // 9
```


```julia
import Base.+
function +(a::MyRational, b::MyRational)
    MyRational(a.n*b.d+b.n*a.d, a.d*b.d)
end

a + b
```


```
83 // 36
```


---






# Générer des données bruitées


```julia
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
```


```
([0.9706563288552144 -1.445177115286233 … 0.6785478622508077 -0.2702750330391049; -0.9792184115351997 2.7074239417157804 … -0.2851921262946706 -0.20563037602802728; … ; -0.03280312924463938 0.759804020007466 … 0.4207068461244445 -1.2139401711703544; -0.6007922233555612 -0.8814369061964817 … -0.31934708892587327 1.0914985820684815], [0.05736712902036197, 0.9054712475117923, 1.2515751259310877, -0.7290190325950813, -0.3844694490639139, 0.8969921297327403, -0.37024047958804807, -0.9253095221977053, -0.5211913459968133, 1.022435013919742  …  0.738820217823834, 0.8959881951955971, -0.09375639249386229, -0.04720334715225621, -0.4403434166322156, -0.812267018858789, -0.6054535072725257, 0.2911042443203091, 0.05570431860650896, 0.42680788523773283])
```


---






## Fonction non typée pour une régression linéaire


```julia
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
```


```
linear_regression (generic function with 1 method)
```


---


```julia
linear_regression( x, y)
```


```
([0.10007923696855399, 0.20122520323798196, 0.297245503370711, 0.39795764373682924, 0.49782554328307793], 0.2004551171717869)
```


```julia
linear_regression( Float32.(x), Float32.(y))
```


```
([0.10007923926323069, 0.20122520079131082, 0.29724550326119636, 0.39795764144188867, 0.49782554489962366], 0.20045511724877926)
```


Avec des flottants simple précision en entrée, on aimerait que tous les calculs soient faits en simple précision


Si le code source de la méthode est bien écrit, le code source et le type concret de tous les arguments sont des informations suffisantes pour que le compilateur puisse déduire le type concret de chaque variable au sein de la fonction. La fonction est alors dite "type stable" et le compilateur Julia produira un code efficace.


Si, pour diverses raisons, le type d'une variable locale ne peut pas être déduit des types des arguments, le compilateur produit, un code machine beaucoup de structures conditionnelles, couvrant toutes les options de ce que le type de chaque variable pourrait être. La perte de performance est souvent significative, facilement d'un facteur 10.


**Évitez les méthodes qui renvoient des variables dont le type dépend de la valeur.**


**Dans la mesure du possible, utilisez des tableaux avec un type d'élément concret.**


---






## Fonction typée pour une régression linéaire


```julia
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
```


```
linear_regression (generic function with 2 methods)
```


---


```julia
linear_regression( x, y)
```


```
([0.10007923696855399, 0.20122520323798196, 0.297245503370711, 0.39795764373682924, 0.49782554328307793], 0.2004551171717869)
```


```julia
linear_regression( Float32.(x), Float32.(y))
```


```
(Float32[0.100079246, 0.20122519, 0.29724553, 0.39795765, 0.49782556], 0.20045516f0)
```


---






## Type composé pour une régression linéaire


```julia
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
```


---


```julia
model = LinearRegression(x, y)

model.weights
```


```
5-element Vector{Float64}:
 0.10007923696855399
 0.20122520323798196
 0.297245503370711
 0.39795764373682924
 0.49782554328307793
```


```julia
model.bias
```


```
0.2004551171717869
```


```julia
predict(model, x) = x' * model.weights .+ model.bias

function Base.show(io :: IO, model :: LinearRegression)
    println(io, "Linear Regression")
    println(io, "=================")
    println(io, "weights : $(round.(model.weights, digits=3))")
    println(io, "bias : $(round(model.bias, digits=3))")
end

model
```


```
Linear Regression
=================
weights : [0.1, 0.201, 0.297, 0.398, 0.498]
bias : 0.2

```


---






# Les particularités du langage Julia


  * Les types composés ne pourront pas être modifiés dans une même session, peu pratique mais offre une meilleure gestion de la mémoire.
  * Toutes les fonctions sont compilées lors de leur premier appel, un peu de latence, mais l'exécution est beaucoup plus rapide ensuite.
  * Les fonctions sont spécialisées suivant le type de leurs arguments, plus de compilation, mais cela améliore la lisibilité de vos interfaces.
  * Julia est interactif, facile à utiliser, mais la création de binaire est compliquée.
  * Julia possède plusieurs manières pour manipuler les tableaux, très pratique pour optimiser son empreinte mémoire, mais nécessite un apprentissage.
  * La bibliothèque standard et quasiment tous les packages sont écrits en Julia, il est facile de s'en inspirer.
  * Il y a un mécanisme de génération de code et de métaprogrammation qui permet de faire de très belles interfaces. Il nécessite des compétences avancées et le déverminage est plus difficile.
  * L'interface avec R et Python est très facile, le C et le Fortran sont également nativement encapsulables. En revanche, le C++ c'est plus compliqué...


---






# Ce qu'il faut savoir avant de se lancer...


  * L'apprentissage profond en Julia est en retard par rapport à ce que proposent les bibliothèques proposées en Python. Il y a tout de même des choses intéressantes en différentiation automatique.
  * Beaucoup de packages disponibles sont non maintenus ou complètement abandonnés. Il faut être prudent lorsque l'on choisit ses dépendances. Si la licence le permet, on peut toujours récupérer des choses dans les packages, ils sont écrits en Julia.
  * Le "workflow" à mi-chemin entre le langage interprété et le langage compilé est parfois déroutant et peut dégoûter les débutants. Les durées de compilations ont été fortement réduites depuis la version 1.
  * L'IDE préconisé est VSCode, mais il a moins de fonctionnalités que ce qui est proposé dans les autres langages.
  * L'optimisation de performance en Julia nécessite un apprentissage particulier. On peut être déçu après une traduction naïve depuis du Fortran ou du MATLAB. C'est aussi la partie la plus "satisfaisante" du langage et attention cela peut devenir addictif :-)
  * Les opérations vectorisées sont moins efficaces en Julia et cela peut être décevant si l'on est attaché à cette manière de coder les algorithmes. En revanche, vous pouvez faire autant de boucles que vous voulez !
  * En Julia, il faut faire des fonctions pour minimiser les calculs et les allocations dans l'environnement global. Les erreurs dues à la portée des variables sont parfois un peu déroutantes.


---






# Pourquoi il faut se lancer !


  * **C'est beau** : une syntaxe lisible très proche des mathématiques
  * **C'est puissant** : un langage bien conçu, le "multiple dispatch" et la possibilité de l'optimiser de manière incrémentale en ajoutant progressivement les types, par exemple, est agréable. Il y a beaucoup de fonctions haut-niveau disponibles qui permettent d'écrire des algorithmes complexes en quelques lignes de codes.
  * **C'est fait pour nous** : c'est un langage fait pour les sciences qui est gouverné par des scientifiques. L'écosystème est déjà très large dans beaucoup de disciplines. Pour les équations différentielles et l'optimisation mathématique, Julia est déjà populaire.
  * **C'est facile** : l'apprentissage est rapide et l'accès au calcul sur GPU, par exemple, est très simple.
  * **C'est interactif** : Le Julia REPL est puissant et très pratique et on peut utiliser les notebooks Jupyter ou Pluto.
  * **C'est rapide** : Le code est compilé, la parallélisation est proposée nativement et très accessible. C'est un candidat tout à fait crédible pour un projet HPC.
  * **C'est ouvert** : Le langage Julia est publié sous une licence MIT. Le système de gestion des packages est très efficace et c'est très simple de créer et proposer son propre package.


---






# Bibliographie


L'article des créateurs:


**Julia: A Fresh Approach to Numerical Computing**


*Jeff Bezanson, Alan Edelman, Stefan Karpinski, Viral B. Shah*  SIAM Rev., 59(1), 65–98. (34 pages) 2012


Sources des images : [Matthijs Cox - My Target Audience](https://scientificcoder.com/my-target-audience#heading-the-two-culture-problem)


[Julia Hub](https://juliahub.com/) : le portail Julia pour exécuter des notebooks et voir tous les packages existants.


Le livre blanc : [Why Julia ?](https://juliahub.com/assets/pdf/why_julia.pdf)


Un article en français : [A la découverte de Julia](https://zestedesavoir.com/articles/pdf/78/a-la-decouverte-de-julia.pdf)


Un support pour se lancer : [Modern Julia Workflows](https://modernjuliaworkflows.github.io)


Liste du CNRS avec la newsletter  https://github.com/pnavaro/NouvellesJulia


Liste rennaise pour les ateliers et les formations https://github.com/pnavaro/math-julia


---


*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
