ENV["GKSwstype"]="100"

using Literate
using Plots
using Remark

slides = Remark.slideshow(@__DIR__,
                          options = Dict("ratio" => "16:9"),
                          title = "CargoChat Julia 2024")
