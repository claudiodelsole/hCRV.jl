# imports
import NPZ: npzwrite, npzread
import Statistics: mean, median, quantile
using Plots

# quantiles
ql, qu = 0.25, 0.75

# plot function
function plot_times(plt::Plots.Plot, values::Vector{Int64}, times::Matrix{Float64}, color::Int64, shape::Symbol)

    # retrieve dimension
    nvals, nruns = size(times)

    # initialize
    mu, lower, upper = zeros(nvals), zeros(nvals), zeros(nvals)

    # compute quantiles
    for id in axes(times, 1)
        idxs = times[id,:] .> 0.0
        if sum(idxs) >= nruns * 0.75
            mu[id], lower[id], upper[id] = median(times[id,idxs]), quantile(times[id,idxs], ql), quantile(times[id,idxs], qu)
        else
            mu[id], lower[id], upper[id] = NaN, NaN, NaN 
        end
    end
    
    # plot
    plot!(plt, values, mu, linecolor = color, markershape = shape, markercolor = color, markersize = (shape == :rect ? 3 : 4), markerstrokewidth = 0)
    plot!(plt, values, lower, fillrange = upper, linealpha = 0.0, fillcolor = color, fillalpha = 0.3)

end

##########

# load times
data = npzread("saves/times_d.npz")
dvalues, num_clusters = data["dvalues"], data["num_clusters"]
time_mcmc, ess_mcmc = data["mcmc"], data["mcmc_ess"]
time_exact, ess_exact = data["exact"], data["exact_ess"]
time_crf, ess_crf = data["crf"], data["crf_ess"]
time_collapsed, ess_collapsed = data["collapsed"], data["collapsed_ess"]

begin # plot times

    # initialize plot
    pltd = plot(legend = false, left_margin = 5Plots.mm, fontfamily = "Computer Modern")

    # Metropolis-Hastings with lognormal proposals
    plot_times(pltd, dvalues, time_mcmc ./ ess_mcmc * 1.0e6, 1, :rect)

    # exact sampler
    plot_times(pltd, dvalues, time_exact ./ ess_exact * 1.0e6, 3, :circle)

    # chinese restaurant process sampler
    plot_times(pltd, dvalues, time_crf ./ ess_crf * 1.0e6, 5, :diamond)

    # collapsed Gibbs sampler
    plot_times(pltd, dvalues, time_collapsed ./ ess_collapsed * 1.0e6, 2, :utriangle)

    # plot attributes
    ylabel!(pltd, "CPU time (µs)")
    ylims!(pltd, 0.0, 100.0)

end

begin # plot clusters

    # initialize plot
    pltdk = plot(legend = false, fontfamily = "Computer Modern")

    # number of clusters
    mu = mean(num_clusters, dims = 2)
    plot!(pltdk, dvalues, mu, linecolor = :black, markershape = :star6, markercolor = :black, markersize = 5, markerstrokewidth = 0)

    # plot attributes
    xlabel!(pltdk, "\$d\$")
    ylabel!(pltdk, "\$k\$")
    ylims!(pltdk, 0.0, 15.0)

end

##########

# load times
data = npzread("saves/times_n.npz")
nvalues, num_clusters = data["nvalues"], data["num_clusters"]
time_mcmc, ess_mcmc = data["mcmc"], data["mcmc_ess"]
time_exact, ess_exact = data["exact"], data["exact_ess"]
time_crf, ess_crf = data["crf"], data["crf_ess"]
time_collapsed, ess_collapsed = data["collapsed"], data["collapsed_ess"]

begin # plot times

    # initialize plot
    pltn = plot(legend = false, fontfamily = "Computer Modern")

    # Metropolis-Hastings with lognormal proposals
    plot_times(pltn, nvalues, time_mcmc ./ ess_mcmc * 1.0e6, 1, :rect)

    # exact sampler
    plot_times(pltn, nvalues, time_exact ./ ess_exact * 1.0e6, 3, :circle)

    # chinese restaurant process sampler
    plot_times(pltn, nvalues, time_crf ./ ess_crf * 1.0e6, 5, :diamond)

    # collapsed Gibbs sampler
    plot_times(pltn, nvalues, time_collapsed ./ ess_collapsed * 1.0e6, 2, :utriangle)

    # plot attributes
    # ylabel!(pltn, "CPU time (µs)")
    ylims!(pltn, 0.0, 100.0)

end

begin # plot clusters

    # initialize plot
    pltnk = plot(legend = false, fontfamily = "Computer Modern")

    # number of clusters
    mu = mean(num_clusters, dims = 2)
    plot!(pltnk, dvalues, mu, linecolor = :black, markershape = :star6, markercolor = :black, markersize = 5, markerstrokewidth = 0)

    # plot attributes
    xlabel!(pltnk, "\$n_i\$")
    # ylabel!(pltnk, "\$k\$")
    ylims!(pltnk, 0.0, 15.0)

end

##########

# combine plots
subplots = plot(pltd, pltn, pltdk, pltnk, layout = @layout([d{0.7h} n; dk nk]))

begin # legend plot

    # create legend
    plegend = plot(axis = false, grid = false, legend = :bottom, legend_column = 5, fontfamily = "Computer Modern")

    # colors and labels
    mycolors, mylabels, myshapes = [1, 3, 5, 2], ["MCMC", "exact", "CRF", "collapsed"], [:rect, :circle, :diamond, :utriangle]

    # legend patches
    for (color, label, shape) in zip(mycolors, mylabels, myshapes)
        plot!(plegend, [NaN], [NaN], label = label, linecolor = color, markershape = shape, markercolor = color, markerstrokewidth = 0)
    end
    plot!(plegend, [NaN], [NaN], label = "clusters", linecolor = :black, markershape = :star6, markercolor = :black, markerstrokewidth = 0)

    # combine legend
    plot(subplots, plegend, layout = @layout([plt; leg{0.05h}]), size = (960, 600))

end 

# save figure
savefig("figures/times.pdf")

##########

# load times
data = npzread("saves/times_k.npz")
kvalues = data["kvalues"]
time_mcmc, ess_mcmc = data["mcmc"], data["mcmc_ess"]
time_exact, ess_exact = data["exact"], data["exact_ess"]
time_crf, ess_crf = data["crf"], data["crf_ess"]
time_collapsed, ess_collapsed = data["collapsed"], data["collapsed_ess"]

begin # plot times

    # initialize plot
    pltk = plot(legend = false, fontfamily = "Computer Modern")

    # Metropolis-Hastings with lognormal proposals
    plot_times(pltk, kvalues, time_mcmc ./ ess_mcmc * 1.0e6, 1, :rect)

    # exact sampler
    plot_times(pltk, kvalues, time_exact ./ ess_exact * 1.0e6, 3, :circle)

    # chinese restaurant process sampler
    # plot_times(pltk, kvalues, time_crf ./ ess_crf * 1.0e6, 5, :diamond)

    # collapsed Gibbs sampler
    plot_times(pltk, kvalues, time_collapsed ./ ess_collapsed * 1.0e6, 2, :utriangle)

    # plot attributes
    xlabel!("\$k\$")
    ylabel!(pltk, "CPU time (µs)")
    ylims!(pltk, 0.0, 40.0)

end

begin # legend plot

    # create legend
    plegend = plot(axis = false, grid = false, legend = :bottom, legend_column = 3, fontfamily = "Computer Modern")

    # colors and labels
    mycolors, mylabels, myshapes = [1, 3, 2], ["MCMC", "exact", "collapsed"], [:rect, :circle, :utriangle]

    # legend patches
    for (color, label, shape) in zip(mycolors, mylabels, myshapes)
        plot!(plegend, [NaN], [NaN], label = label, linecolor = color, markershape = shape, markercolor = color, markerstrokewidth = 0)
    end

    # combine legend
    plot(pltk, plegend, layout = @layout([plt; leg{0.03h}]), size = (480, 420))

end 

# save figure
savefig("figures/times_k.pdf")
