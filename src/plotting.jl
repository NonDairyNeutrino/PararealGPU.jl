# lines
plot(
    discretizedDomain, 
    [solution, exp.(discretizedDomain)],
    label = ["numeric" "analytic"],
    title = "coarse: $COARSEDISCRETIZATION, fine: $FINEDISCRETIZATION"
)
# Dots to highlight numeric solution
scatter!(discretizedDomain, solution, label = "")
# subdomain coarse propagation
# for region in eachindex(subDomainCoarse)
#     scatter!(
#         subDomainCoarse[region]...,
#         label = "Subdomain Coarse $region",
#         markershape = :rect
#     )
# end
# # subdomain fine propagation
# for region in eachindex(subDomainFine)
#     scatter!(
#         subDomainFine[region]...,
#         label = "Subdomain Fine $region",
#         markershape = :diamond
#     ) 
# end
# display(plot!())
# Correctors
# scatter!(discretizedDomain, subDomainCorrectors, label = "Correctors") # Correctors
# plot!(corrected..., label = "Corrected")
