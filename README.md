# CargoChat Julia 5 avril 2024

[diapositives](https://pnavaro.github.io/CargoChat2024)

Supports de présentation pour le [Cargo Chat](https://cargo.resinfo.org/spip.php?article106) du 5 avril 2024

## Résumé

Julia est un langage de programmation créé au MIT en 2012. Sa
véritable naissance comme solution intéressante pour vos projets
de développement est juillet 2018 avec la sortie de la version 1.0
durant la conférence annuelle JuliaCon qui a eu lieu à Londres.

Julia a un positionnement particulier parmi les langages existants,
car il va se placer entre les langages interprétés (Python-R) et
les langages compilés (Fortran-C/C++). Il a beaucoup de similarités
avec MATLAB et aura les mêmes cibles d'usage.

Son objectif est d'offrir les avantages de Python-R avec un grand
nombre de bibliothèques disponibles (packages) qui permettent une
programmation haut-niveau. Vous allez pouvoir programmer rapidement
votre algorithme avec un effort contenu et peu de lignes de code.
Vous aurez également accès directement à la visualisation sans
passer par des logiciels extérieurs. Vous pourrez facilement partager
vos travaux sous forme de packages comme les deux autres langages
précités.

Julia est aussi un langage compilé donc vous pouvez accéder aux
mêmes performances que les langages Fortran-C/C++. Avec Python-R
si vous souhaitez obtenir ces performances vous devez "encapsuler"
du C ou du Fortran dans vos programmes. Les concepteurs de packages
utilisent donc souvent deux langages. Cette technique est facilitée
par de nombreux outils (Cython, Rcpp,...), mais elle crée une
différence importante entre ceux qui conçoivent les packages et
ceux qui les utilisent. Avec Julia ce "problème des deux langages"
est supprimé et il faudra apprendre à optimiser son code Julia
plutôt qu'en apprendre un autre.

Je vous propose pour cette rencontre de voir ensemble quelques
exemples de codes Julia pour découvrir sa syntaxe. Nous verrons que
Julia a des avantages, mais aussi des inconvénients. C'est un projet
passionnant, car il a été conçu par des scientifiques pour faire
des sciences et c'est ce qui le rend si attractif pour nos usages
quotidiens.

## Références

- [Why We Created Julia](https://julialang.org/blog/2012/02/why-we-created-julia/)
- [My Target Audience](https://scientificcoder.com/my-target-audience#heading-the-two-culture-problem)
- [How to solve the two language problem?](https://scientificcoder.com/how-to-solve-the-two-language-problem)
