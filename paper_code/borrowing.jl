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
using Plots

# sampling models
include("../aux_code/models.jl")
include("../aux_code/functionals.jl")

# set colors
# mycolors = ["#abce94", "#d37e61", "#4d7baf"]
mycolors = [1, 2, 3]

# compute parameter values
function params_equation(alpha::Float64, alpha0::Float64, rho::Float64)::Float64

    return 1.0 - rho * (1.0 + alpha0 / alpha * exp(1.0  / alpha) * expint(alpha0, 1.0 / alpha))

end # params_equation

# set seed
seed!(110590)

# parameter values
rhos = [0.1, 0.3, 0.5, 0.7, 0.9]
sigma2s = [0.1, 0.3, 0.5, 0.7, 0.9]

# initialize parameters
alphas = zeros(length(rhos), length(sigma2s))
alpha0s = zeros(length(rhos), length(sigma2s))

# loop on parameters
for (idx, rho) in enumerate(rhos)
    for (idy, sigma2) in enumerate(sigma2s)

        # compute parameters
        alpha0s[idx,idy] = 1.0 / (rho * sigma2) - 1.0
        alphas[idx,idy] = find_zero(alpha::Float64 -> params_equation(alpha, alpha0s[idx, idy], rho), 
                                    (1.0e-2, 1.0e2), Bisection(); atol = 1.0e-8, maxevals = 100)

    end
end

# parameters
d = 3
means = [4.0, 3.0, 2.0]
counts_group = fill(10, d)

# sample observations
X = model_poisson(counts_group, means, seed = 180396)
# X = [[4, 2, 4, 4, 5, 5, 7, 3, 2, 5], 
    #  [1, 5, 2, 1, 0, 5, 3, 2, 3, 5], 
    #  [2, 2, 1, 2, 1, 1, 2, 2, 3, 3]]

# empirical means
empirical_means = [mean(X_group) for X_group in X]

# model parameters
b, b0 = 1.0, 1.0
ell = 50

# prior parameters
mu_prior, std_prior = 10.0, 1.0

# initialize output
posterior_means = zeros(length(rhos), length(sigma2s), d)

# number of samples
num_samples = 10000

# compute base samples
base_samples = repeat(rand(Normal(mu_prior, std_prior), ell, 1, num_samples), 1, d, 1)

# loop on parameters
for (idx, rho) in enumerate(rhos)
    for (idy, sigma2) in enumerate(sigma2s)

        # retrieve parameters
        alpha0, alpha = alpha0s[idx,idy], alphas[idx,idy]

        # sample posterior
        probs, probsc, _, Xstar, _ = posterior_gamma_exact(X, alpha0, b0 / alpha, b, num_samples, L = ell, normalize = true)

        # compute random means
        rmeans = random_means(probs, probsc, Float64.(Xstar), base_samples)

        # record estimates
        posterior_means[idx,idy,:] = mean(rmeans, dims = 2) 

    end
end

# save times
# npzwrite("saves/borrowing.npz", Dict("rhos" => rhos, "sigma2s" => sigma2s,
#                                      "empirical_means" => empirical_means,
#                                      "posterior_means" => posterior_means))

# create subplots
subplots = plot(layout = (2, 3), legend = false, fontfamily = "Computer Modern")

# loop on variances
for (idy, sigma2) in enumerate(sigma2s)
    
    if idy % 2 == 0 continue end

    # retrieve plot
    plt = subplots[(idy + 1) ÷ 2]
    title!(plt, "\$\\sigma^2 = " * string(sigma2) * "\$")
    
    # empirical means
    for (value, color) in zip(empirical_means, mycolors)
        hline!(plt, [value], linecolor = color, linestyle = :dashdot)
    end
    
    # posterior means
    for (i, color) in enumerate(mycolors)
        plot!(plt, rhos, posterior_means[:,idy,i], linecolor = color, markershape = :rect, markercolor = color, markerstrokewidth = 0)
    end
    
    # plot attributes
    xlabel!(plt, "\$\\rho\$")
    xticks!(plt, rhos)
    xlims!(plt, 0.01, 0.99)
    ylims!(plt, 1.6, 6.4)

end

# loop on correlations
for (idx, rho) in enumerate(rhos)
    
    if idx % 2 == 0 continue end

    # retrieve plot
    plt = subplots[(idx + 1) ÷ 2 + 3]
    
    # empirical means
    for (value, color) in zip(empirical_means, mycolors)
        hline!(plt, [value], linecolor = color, linestyle = :dashdot)
    end
    
    # posterior means
    for (i, color) in enumerate(mycolors)
        plot!(plt, sigma2s, posterior_means[idx,:,i], linecolor = color, markershape = :rect, markercolor = color, markerstrokewidth = 0)
    end
    
    # plot attributes
    title!(plt, "\$\\rho = " * string(rho) * "\$")
    xlabel!(plt, "\$\\sigma^2\$")
    xticks!(plt, sigma2s)
    xlims!(plt, 0.01, 0.99)
    ylims!(plt, 1.6, 6.4)

end

begin # legend plot

    # create legend
    plegend = plot(axis = false, grid = false, legend = :bottom, legend_column = d + 2, fontfamily = "Computer Modern")

    # legend patches
    for (i, color) in enumerate(mycolors)
        plot!(plegend, [NaN], [NaN], seriestype = :bar, label = "group " * string(i), fillcolor = color, linewidth = 0)
    end

    # legend lines
    plot!(plegend, [NaN], [NaN], label = "posterior", linecolor = :black)
    plot!(plegend, [NaN], [NaN], label = "empirical", linecolor = :black, linestyle = :dashdot)

    # combine legend
    plot(subplots, plegend, layout = @layout([plt; leg{0.06h}]), size = (960, 540))

end 

# save figure
savefig("figures/borrowing.pdf")
