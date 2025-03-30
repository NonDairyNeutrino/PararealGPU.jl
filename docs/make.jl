push!(LOAD_PATH,"../src/")

using Documenter
using PararealGPU

makedocs(
    sitename = "PararealGPU.jl",
    format = Documenter.HTML(),
    modules = [PararealGPU],
    pages = [
        "Home"   => "index.md",
        "Manual" => [
            "Guide" => "man/guide.md",
            "Examples" => [
                "man/examples/wave.md"
            ],
            "Parareal" => [
                "man/parareal/parareal.md",
                "man/parareal/subproblems.md",
                "man/parareal/propagation.md",
                "man/parareal/correction.md"
            ],
            "HPC" => [
                "man/hpc/gpu.md",
                "man/hpc/distributed.md"
            ],
            "Numerical Analysis" => [
                "man/analysis/convergence.md",
                "man/analysis/stability.md",
                "man/analysis/energy.md"
            ]
        ],
        "Showcase" => [
            "showcase/particle_production.md"
        ],
        "Developers" => [
            "dev/contributing.md"
        ],
        "release-notes.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/nondairyneutrino/pararealgpu.jl.git"
)
