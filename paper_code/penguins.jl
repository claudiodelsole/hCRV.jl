# load modules
using hCRV
include("../src/HDP.jl")
using .HDP

# imports
import NPZ: npzwrite
import SpecialFunctions: expint
import Distributions: Normal
import Roots: find_zero, Bisection
import Statistics: mean
import Random: seed!
import CSV: CSV
import DataFrames: DataFrame
using Plots

# functionals
include("../aux_code/functionals.jl")

# set colors
# mycolors = ["#abce94", "#d37e61", "#4d7baf"]
mycolors = [1, 2, 3]

##### Penguins dataset

# load dataset
penguins = CSV.File("data/penguins.csv")

# drop missing values
penguins = filter(row -> sum(ismissing.(row)) == 0, penguins)

# filter female penguins
penguins = filter(row -> row.sex == "FEMALE", penguins)

# create data frame
penguins = DataFrame(penguins)

# save dataset
# CSV.write("data/penguins_cleaned.csv", penguins)

# retrieve observed variables
species = String.(penguins.species)
lengths = penguins.flipper_length_mm

# retrieve dimension
d = length(unique(species))

# create observations structure
X = Vector{Vector{Int}}(undef, d)
for (i, name) in enumerate(unique(species))
    idxs = species .== name
    X[i] = lengths[idxs]
end

# observations per group
println("Number of observations (n): ", length(vcat(X...))); println()
println("Number of groups (d): ", d); println()
println("Observations per group:")
for (i, name) in enumerate(unique(species))
    println(name, ": ", length(X[i]))
end

# number of clusters
println(); println("Number of clusters (k): ", length(unique(vcat(X...))))

# empirical means
empirical_means = [mean(X_group) for X_group in X]

# model parameters
b, b0 = 1.0, 1.0
ell = 50

# prior parameters
mu_prior, std_prior = 100.0, 10.0

# number of samples
num_samples = 10000
burnin = 1000

##### Same parameters

# set seed
seed!(180396)

# parameters values
alpha0s = [0.5, 2.0, 5.0, 12.0, 20.0]
alpha = 1.0

# initialize outputs
posterior_means_hcrv = zeros(d, length(alpha0s))
posterior_means_hdp = zeros(d, length(alpha0s))

# compute base samples
base_samples = repeat(rand(Normal(mu_prior, std_prior), ell, 1, num_samples), 1, d, 1)

# loop on parameters
for (idx, alpha0) in enumerate(alpha0s)

    ### hCRV: exact sampler
    probs, probsc, _, Xstar, _ = posterior_gamma_exact(X, alpha0, b0 / alpha, b, num_samples, L = ell, normalize = true)

    # compute random means
    rmeans = random_means(probs, probsc, Float64.(Xstar), base_samples)

    # record estimates
    posterior_means_hcrv[:,idx] = mean(rmeans, dims = 2)

    ### HDP: collapsed Gibbs sampler
    probs, probsc, _, Xstar, _ = posterior_hdp(X, alpha0, alpha, num_samples, burnin = burnin, L = ell, prior = false, collapsed = true)

    # compute random means
    rmeans = random_means(probs, probsc, Float64.(Xstar), base_samples)

    # record estimates
    posterior_means_hdp[:,idx] = mean(rmeans, dims = 2)

end

# save means
# npzwrite("saves/penguins_params.npz", 
#          Dict("alpha" => alpha, "alpha0s" => alpha0s, "empirical_means" => empirical_means,
#               "posterior_means_hcrv" => posterior_means_hcrv, "posterior_means_hdp" => posterior_means_hdp))

begin # plot

    # initialize plot
    plparams = plot(legend = false, fontfamily = "Computer Modern")
    title!(plparams, "\$\\alpha = " * string(alpha) * "\$")

    # loop on groups
    for (i, color) in enumerate(mycolors)

        # posterior means
        plot!(plparams, alpha0s, posterior_means_hcrv[i,:], linecolor = color, markershape = :rect, markercolor = color, markerstrokewidth = 0)
        plot!(plparams, alpha0s, posterior_means_hdp[i,:], linecolor = color, linestyle = :dash, markershape = :diamond, markercolor = color, markersize = 5, markerstrokewidth = 0)

        # empirical means
        hline!(plparams, [empirical_means[i]], linecolor = color, linestyle = :dashdot, linealpha = 0.75)

    end

    # plot attributes
    xlabel!(plparams, "\$\\alpha_0\$")
    xlims!(plparams, 0.0, 22.0)
    ylims!(plparams, 180.0, 220.0)

end

##### Same correlation

# compute parameter values
function params_equation(alpha::Float64, alpha0::Float64, rho::Float64)::Float64

    return 1.0 - rho * (1.0 + alpha0 / alpha * exp(1.0  / alpha) * expint(alpha0, 1.0 / alpha))

end # params_equation

# parameter values
rhos = [0.2, 0.4, 0.6, 0.8]
sigma2_hcrv, sigma2_hdp = 0.2, 0.8

# initialize outputs
posterior_means_hcrv = zeros(d, length(rhos))
posterior_means_hdp = zeros(d, length(rhos))

# initialize parameters
alphas = zeros(length(rhos))
alpha0s = zeros(length(rhos))

# loop on parameters
for (idx, rho) in enumerate(rhos)

    # ### hCRV: compute parameters
    alpha0 = 1.0 / (rho * sigma2_hcrv) - 1.0
    local alpha = find_zero(alpha::Float64 -> params_equation(alpha, alpha0, rho), 
                                    (1.0e-2, 1.0e2), Bisection(); atol = 1.0e-8, maxevals = 100)

    # exact sampler
    probs, probsc, _, Xstar, _ = posterior_gamma_exact(X, alpha0, b0 / alpha, b, num_samples, L = ell, normalize = true)

    # compute random means
    rmeans = random_means(probs, probsc, Float64.(Xstar), base_samples)

    # record estimates
    posterior_means_hcrv[:,idx] = mean(rmeans, dims = 2)

    # HDP: compute parameters
    alpha0 = 1.0 / (rho * sigma2_hdp) - 1.0
    alpha = (1.0 / sigma2_hdp - 1.0) / (1.0 - rho)

    # collapsed Gibbs sampler
    probs, probsc, _, Xstar, _ = posterior_hdp(X, alpha0, alpha, num_samples, burnin = burnin, L = ell, prior = false, collapsed = true)

    # compute random means
    rmeans = random_means(probs, probsc, Float64.(Xstar), base_samples)

    # record estimates
    posterior_means_hdp[:,idx] = mean(rmeans, dims = 2)

end

# save means
# npzwrite("saves/penguins_correlation.npz", 
#          Dict("rhos" => rhos, "empirical_means" => empirical_means,
#               "posterior_means_hcrv" => posterior_means_hcrv, "posterior_means_hdp" => posterior_means_hdp))

begin # plot

    # initialize plot
    plcorrelation = plot(legend = false, fontfamily = "Computer Modern")
    title!("\$\\sigma^2_{hCRV} \\neq \\sigma^2_{HDP}\$")

    # loop on groups
    for (i, color) in enumerate(mycolors)

        # posterior means
        plot!(plcorrelation, rhos, posterior_means_hcrv[i,:], linecolor = color, markershape = :rect, markercolor = color, markerstrokewidth = 0)
        plot!(plcorrelation, rhos, posterior_means_hdp[i,:], linecolor = color, linestyle = :dash, markershape = :diamond, markercolor = color, markersize = 5, markerstrokewidth = 0)

        # empirical means
        hline!(plcorrelation, [empirical_means[i]], linecolor = color, linestyle = :dashdot, linealpha = 0.75)

    end

    # plot attributes
    xlabel!(plcorrelation, "\$\\rho\$")
    xlims!(plcorrelation, 0.1, 0.9)
    ylims!(plcorrelation, 180.0, 220.0)

end

##### Fair comparison

# parameter values
rhos = [0.2, 0.4, 0.6, 0.8]
sigma2 = 0.5

# initialize outputs
posterior_means_hcrv = zeros(d, length(rhos))
posterior_means_hdp = zeros(d, length(rhos))

# initialize parameters
alphas = zeros(length(rhos))
alpha0s = zeros(length(rhos))

# loop on parameters
for (idx, rho) in enumerate(rhos)

    # ### hCRV: compute parameters
    alpha0 = 1.0 / (rho * sigma2) - 1.0
    local alpha = find_zero(alpha::Float64 -> params_equation(alpha, alpha0, rho), 
                                    (1.0e-2, 1.0e2), Bisection(); atol = 1.0e-8, maxevals = 100)

    # exact sampler
    probs, probsc, _, Xstar, _ = posterior_gamma_exact(X, alpha0, b0 / alpha, b, num_samples, L = ell, normalize = true)

    # compute random means
    rmeans = random_means(probs, probsc, Float64.(Xstar), base_samples)

    # record estimates
    posterior_means_hcrv[:,idx] = mean(rmeans, dims = 2)

    # HDP: compute parameters
    alpha0 = 1.0 / (rho * sigma2_hdp) - 1.0
    alpha = (1.0 / sigma2 - 1.0) / (1.0 - rho)

    # collapsed Gibbs sampler
    probs, probsc, _, Xstar, _ = posterior_hdp(X, alpha0, alpha, num_samples, burnin = burnin, L = ell, prior = false, collapsed = true)

    # compute random means
    rmeans = random_means(probs, probsc, Float64.(Xstar), base_samples)

    # record estimates
    posterior_means_hdp[:,idx] = mean(rmeans, dims = 2)

end

# save means
# npzwrite("saves/penguins_moments.npz", 
#          Dict("sigma2" => sigma2, "rhos" => rhos, "empirical_means" => empirical_means,
#               "posterior_means_hcrv" => posterior_means_hcrv, "posterior_means_hdp" => posterior_means_hdp))

begin # plot

    # initialize plot
    plmoments = plot(legend = false, fontfamily = "Computer Modern")
    title!("\$\\sigma^2 = " * string(sigma2) * "\$")

    # loop on groups
    for (i, color) in enumerate(mycolors)

        # posterior means
        plot!(plmoments, rhos, posterior_means_hcrv[i,:], linecolor = color, markershape = :rect, markercolor = color, markerstrokewidth = 0)
        plot!(plmoments, rhos, posterior_means_hdp[i,:], linecolor = color, linestyle = :dash, markershape = :diamond, markercolor = color, markersize = 5, markerstrokewidth = 0)

        # empirical means
        hline!(plmoments, [empirical_means[i]], linecolor = color, linestyle = :dashdot, linealpha = 0.75)

    end

    # plot attributes
    xlabel!(plmoments, "\$\\rho\$")
    xlims!(plmoments, 0.1, 0.9)
    ylims!(plmoments, 180.0, 220.0)

end

# combine plots
subplots = plot(plparams, plcorrelation, plmoments, layout = (1, 3))

begin # legend plot

    # create legend
    plegend = plot(axis = false, grid = false, legend = :bottom, legend_column = d + 3, fontfamily = "Computer Modern")

    # legend patches
    for (name, color) in zip(unique(species), mycolors)
        plot!(plegend, [NaN], [NaN], seriestype = :bar, label = name, fillcolor = color, linewidth = 0)
    end

    # legend lines
    plot!(plegend, [NaN], [NaN], label = "hCRV", linecolor = :black, markershape = :rect, markercolor = :black)
    plot!(plegend, [NaN], [NaN], label = "HDP", linecolor = :black, linestyle = :dash, markershape = :diamond, markercolor = :black)
    plot!(plegend, [NaN], [NaN], label = "empirical", linecolor = :black, linestyle = :dashdot)

    # combine legend
    plot(subplots, plegend, layout = @layout([plt; leg{0.15h}]), size = (960, 300))

end 

# save figure
savefig("figures/penguins.pdf")
