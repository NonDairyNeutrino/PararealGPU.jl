using Plots: plot, plot!, savefig
# plot true solution
plot(
    TRUEDOMAIN,
    [TRUEPOSITION, TRUEVELOCITY],
    label = ["position - true" "velocity - true"]
)
plot!(
    solution.domain,
    [solution.positionSequence .|> only, solution.velocitySequence .|> only],
    label = ["position" "velocity"],
    title = "coarse: $COARSEDISCRETIZATION, fine: $FINEDISCRETIZATION"
)
plot_dir  = string(pwd(), "/examples/images/")
plot_name = "wave_single_cd$(COARSEDISCRETIZATION)_fd$(FINEDISCRETIZATION)_ub$(DOMAINUPPERBOUND/pi)pi.pdf"
plot_abs_path = plot_dir * plot_name
println("Plot saved at ", plot_abs_path)
savefig(plot_abs_path)
run(`codium $plot_abs_path`)