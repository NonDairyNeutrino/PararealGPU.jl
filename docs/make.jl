using Documenter
using Parareal

makedocs(
    sitename = "Parareal",
    format = Documenter.HTML(),
    modules = [Parareal]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
