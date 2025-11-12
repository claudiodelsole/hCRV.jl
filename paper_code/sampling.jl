# load modules
using hCRV
include("../src/HDP.jl")
using .HDP

# imports
import MCMCDiagnosticTools: ess
import Statistics: quantile
import KernelDensity: kde
using Plots

# sampling models
include("../aux_code/models.jl")

# parameters
d = 4
means = [2.0, 3.0, 4.0, 5.0]
counts_group = fill(50, d)

# sample observations
X = model_poisson(counts_group, means, seed = 110590)
# X = model_hdp(counts_group, 3.0, 5.0, seed = 110590)

# extract info from data
counts, Xstar = hCRV.setup_hcrv(X)
d, k = size(counts)

# model parameters
alpha, b = 1.0, 1.0
alpha0, b0 = 1.0, 1.0

# number of samples
num_samples = 10000
burnin = 1000

##### random measures

# Metropolis-Hastings with gamma proposals
# jumps_mcmc, jumpsc_mcmc, counts, Xstar, dgn_mcmc = posterior_gamma_mcmc(X, alpha0, b0 / alpha, b, num_samples, burnin = burnin, logscale = false)
# @profview posterior_gamma_mcmc(X, alpha0, b0 / alpha, b, num_samples, burnin = 0, logscale = false)

# Metropolis-Hastings with lognormal proposals
jumps_mcmc, jumpsc_mcmc, counts, Xstar, dgn_mcmc = posterior_gamma_mcmc(X, alpha0, b0 / alpha, b, num_samples, burnin = burnin, logscale = true)
# @profview posterior_gamma_mcmc(X, alpha0, b0 / alpha, b, num_samples, burnin = 0, logscale = true)

# exact sampler
jumps_exact, jumpsc_exact, counts, Xstar, dgn_exact = posterior_gamma_exact(X, alpha0, b0 / alpha, b, num_samples)
# @profview posterior_gamma_exact(X, alpha0, b0 / alpha, b, num_samples)

# initialize plots
plots = Vector{Plots.Plot}(undef, k)

# loop on clusters
for j in eachindex(Xstar)

    # print title
    println("----------")
    println("Jumps -- Value: ", Xstar[j], "; Counts: ", sum(counts[:,j])); println()

    # create subplots
    subplots = plot(layout = (1, d), fontfamily = "Computer Modern", left_margin = 5Plots.mm, bottom_margin = 2Plots.mm, legend = false)

    # plot limits
    maxval = maximum([quantile(jumps_exact[j,i,:], 0.99) for i in range(1, d)])

    for i in range(1, d)

        # retrieve plot
        plt = subplots[i]

        # Metropolis-Hastings with lognormal proposals
        kdest = kde(jumps_mcmc[j,i,:], boundary = (0.0, 2.0 * maxval))
        plot!(plt, kdest.x, kdest.density, linecolor = 1)

        # exact sampler
        kdest = kde(jumps_exact[j,i,:], boundary = (0.0, 2.0 * maxval))
        plot!(plt, kdest.x, kdest.density, linecolor = 2)

        # annotate counts
        ymax = 1.2 * min(maximum(kdest.density), 10.0 / maxval)
        annotate!(plt, (maxval, 0.95 * ymax, ("counts: " * string(counts[i,j]), :right, 8)))

        # plot attributes
        xlabel!(plt, "\$J_{" *string(i) * "," * string(j) * "}\$")
        if i == 1 ylabel!(plt, "density") end
        xlims!(plt, 0.0, maxval)
        ylims!(plt, 0.0, ymax)

    end

    # effective sample sizes
    println("MCMC:\t", join([string(ess(jumps_mcmc[j,i,:])) for i in range(1, d)], "\t"))
    println("exact:\t", join([string(ess(jumps_exact[j,i,:])) for i in range(1, d)], "\t"))
    println()

    # save plot
    plots[j] = subplots

end

begin # legend plot

    # create legend
    plegend = plot(axis = false, grid = false, legend = :bottom, legend_column = 2, fontfamily = "Computer Modern")

    # legend lines
    plot!(plegend, [NaN], [NaN], label = "MCMC", linecolor = 1)
    plot!(plegend, [NaN], [NaN], label = "exact", linecolor = 2)

    # combine legend
    plot(plots[5], plots[8], plegend, layout = @layout([plt1; plt2; leg{0.1h}]), size = (1200, 500))

end 

# save figure
savefig("figures/posterior.pdf")

# minimum ess
println("----------")
println("Minimum ESS:\t", minimum([ess(jumps_mcmc[j,i,:]) for j in eachindex(Xstar), i in range(1, d)])); println()

##### normalized random measures

# Metropolis-Hastings with gamma proposals
# probs_mcmc, probsc_mcmc, counts, Xstar, dgn_mcmc = posterior_gamma_mcmc(X, alpha0, b0 / alpha, b, num_samples, burnin = burnin, logscale = false, normalize = true)

# Metropolis-Hastings with lognormal proposals
probs_mcmc, probsc_mcmc, counts, Xstar, dgn_mcmc = posterior_gamma_mcmc(X, alpha0, b0 / alpha, b, num_samples, burnin = burnin, logscale = true, normalize = true)

# exact sampler
probs_exact, probsc_exact, counts, Xstar, dgn_exact = posterior_gamma_exact(X, alpha0, b0 / alpha, b, num_samples, normalize = true)

# restaurant franchise sampler
probs_hdp, probsc_hdp, counts, Xstar, dgn_hdp = posterior_hdp(X, alpha0, alpha, num_samples, burnin = burnin)
# @profview posterior_hdp(X, alpha0, alpha, num_samples, burnin = burnin)

# restaurant franchise sampler with prior concentration
probs_hdppr, probsc_hdppr, counts, Xstar, dgn_hdppr = posterior_hdp(X, alpha0, alpha, num_samples, burnin = burnin, prior = true)
# @profview posterior_hdp(X, alpha0, alpha, num_samples, burnin = burnin, prior = true)

# collapsed Gibbs sampler
# probs_hdp, probsc_hdp, counts, Xstar, dgn_hdp = posterior_hdp(X, alpha0, alpha, num_samples, burnin = burnin, collapsed = true)
# @profview posterior_hdp(X, alpha0, alpha, num_samples, burnin = burnin, collapsed = true)

# collapsed Gibbs sampler with prior concentration
# probs_hdppr, probsc_hdppr, counts, Xstar, dgn_hdppr = posterior_hdp(X, alpha0, alpha, num_samples, burnin = burnin, prior = true, collapsed = true)
# @profview posterior_hdp(X, alpha0, alpha, num_samples, burnin = burnin, prior = true, collapsed = true)

# initialize plots
plots = Vector{Plots.Plot}(undef, k)

# loop on clusters
for j in eachindex(Xstar)

    # print title
    println("----------")
    println("Probabilities -- Value: ", Xstar[j], "; Counts: ", sum(counts[:,j])); println()

    # create subplots
    subplots = plot(layout = (1, d), fontfamily = "Computer Modern", left_margin = 5Plots.mm, bottom_margin = 2Plots.mm, legend = false)

    # plot limits
    maxval = maximum([quantile(probs_exact[j,i,:], 0.99) for i in range(1, d)])

    for i in range(1, d)

        # retrieve plot
        plt = subplots[i]

        # Metropolis-Hastings with lognormal proposals
        kdest = kde(probs_mcmc[j,i,:], boundary = (0.0, 2.0 * maxval))
        plot!(plt, kdest.x, kdest.density, linecolor = 1)

        # exact sampler
        kdest = kde(probs_exact[j,i,:], boundary = (0.0, 2.0 * maxval))
        plot!(plt, kdest.x, kdest.density, linecolor = 2)

        # collapsed Gibbs sampler with prior concentration
        kdest = kde(probs_hdppr[j,i,:], boundary = (0.0, 2.0 * maxval))
        plot!(plt, kdest.x, kdest.density, linecolor = 3)

        # collapsed Gibbs sampler
        kdest = kde(probs_hdp[j,i,:], boundary = (0.0, 2.0 * maxval))
        plot!(plt, kdest.x, kdest.density, linecolor = :black, linestyle = :dash)

        # annotate counts
        ymax = 1.2 * min(maximum(kdest.density), 10.0 / maxval)
        annotate!(plt, (maxval, 0.95 * ymax, ("counts: " * string(counts[i,j]), :right, 8)))

        # frequencies
        scatter!(plt, [counts[i,j] / counts_group[i]], [0.02 * ymax], markershape = :star6, markercolor = :red, markerstrokewidth = 0)
        
        # plot attributes
        xlabel!(plt, "\$\\pi_{" *string(i) * "," * string(j) * "}\$")
        if i == 1 ylabel!(plt, "density") end
        xlims!(plt, 0.0, maxval)
        ylims!(plt, 0.0, ymax)

    end

    # effective sample sizes
    println("MCMC:\t", join([string(ess(probs_mcmc[j,i,:])) for i in range(1, d)], "\t"))
    println("exact:\t", join([string(ess(probs_exact[j,i,:])) for i in range(1, d)], "\t"))
    println("HDPpr:\t", join([string(ess(probs_hdppr[j,i,:])) for i in range(1, d)], "\t"))
    println("HDP:\t", join([string(ess(probs_hdp[j,i,:])) for i in range(1, d)], "\t"))
    println()

    # save plot
    plots[j] = subplots

end

begin # legend plot

    # create legend
    plegend = plot(axis = false, grid = false, legend = :bottom, legend_column = 5, fontfamily = "Computer Modern")

    # legend lines
    plot!(plegend, [NaN], [NaN], label = "MCMC", linecolor = 1)
    plot!(plegend, [NaN], [NaN], label = "exact", linecolor = 2)
    plot!(plegend, [NaN], [NaN], label = "HDPpr", linecolor = 3)
    plot!(plegend, [NaN], [NaN], label = "HDP", linecolor = :black, linestyle = :dash)
    scatter!(plegend, [NaN], [NaN], label = "frequency", markershape = :star6, markercolor = :red, markerstrokewidth = 0)

    # combine legend
    plot(plots[5], plots[8], plegend, layout = @layout([plt1; plt2; leg{0.08h}]), size = (1200, 500))

end 

# save figure
savefig("figures/posterior_norm.pdf")

# minimum ess
println("----------")
println("Minimum ESS:\t", minimum([ess(probs_mcmc[j,i,:]) for j in eachindex(Xstar), i in range(1, d)]))
