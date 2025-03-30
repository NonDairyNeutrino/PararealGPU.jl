push!(LOAD_PATH,"../src/")

using Documenter
using PararealGPU

makedocs(
    sitename = "PararealGPU",
    format = Documenter.HTML(),
    modules = [PararealGPU]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
