# imports
import SpecialFunctions: expint
import Roots: find_zero, Bisection
using Plots

# set styles
mystyles = [:dashdot, :solid, :dash]

##### Hierarchical gamma

# compute parameter values
function params_equation(alpha::Float64, alpha0::Float64, rho::Float64)::Float64

    return 1.0 - rho * (1.0 + alpha0 / alpha * exp(1.0  / alpha) * expint(alpha0, 1.0 / alpha))

end # params_equation

# create subplots
subplots = plot(layout = (1, 2), fontfamily = "Computer Modern")

# parameter values
rhos = [0.2, 0.5, 0.8]
sigma2s = range(0.18, 0.82, length = 100)

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

begin # first subplot

    # retrieve plot
    plt = subplots[1]

    # loop on variances
    for (idx, rho) in enumerate(rhos)
        plot!(plt, sigma2s, alpha0s[idx,:], color = 3, linestyle = mystyles[idx], primary = false)
        plot!(plt, sigma2s, alphas[idx,:], color = 2, linestyle = mystyles[idx], primary = false)
    end

    # legend lines
    for (rho, style) in zip(rhos, mystyles)
        plot!(plt, [NaN], [NaN], label = "\$\\rho = " * string(rho) * "\$", linecolor = :black, linestyle = style)
    end

    # plot attributes
    xlabel!(plt, "\$\\sigma^2\$")
    ylabel!(plt, "parameters")
    ylims!(plt, 0.0, 25.0)

end

# parameter values
sigma2s = [0.2, 0.5, 0.8]
rhos = range(0.18, 0.82, length = 100)

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

begin # second subplot

    # retrieve plot
    plt = subplots[2]

    # loop on variances
    for (idy, sigma2) in enumerate(sigma2s)
        plot!(plt, rhos, alpha0s[:,idy], color = 3, linestyle = mystyles[idy], primary = false)
        plot!(plt, rhos, alphas[:,idy], color = 2, linestyle = mystyles[idy], primary = false)
    end

    # legend lines
    for (sigma2, style) in zip(sigma2s, mystyles)
        plot!(plt, [NaN], [NaN], label = "\$\\sigma^2 = " * string(sigma2) * "\$", linecolor = :black, linestyle = style)
    end

    # plot attributes
    xlabel!(plt, "\$\\rho\$")
    # ylabel!(plt, "parameters")
    ylims!(plt, 0.0, 25.0)

end

begin # legend plot

    # create legend
    plegend = plot(axis = false, grid = false, legend = :bottom, legend_column = 2, fontfamily = "Computer Modern")

    # legend patches
    plot!(plegend, [NaN], [NaN], seriestype = :bar, label = "\$\\alpha_0\$", fillcolor = 3, linewidth = 0)
    plot!(plegend, [NaN], [NaN], seriestype = :bar, label = "\$\\alpha\$", fillcolor = 2, linewidth = 0)

    # combine legend
    plot(subplots, plegend, layout = @layout([plt; leg{0.1h}]), size = (960, 420))

end 

# save figure
savefig("figures/moments_hcrv.pdf")

##### Hierarchical Dirichlet process

# create subplots
subplots = plot(layout = (1, 2), fontfamily = "Computer Modern")

# parameter values
rhos = [0.2, 0.5, 0.8]
sigma2s = range(0.18, 0.82, length = 100)

# initialize parameters
alphas = zeros(length(rhos), length(sigma2s))
alpha0s = zeros(length(rhos), length(sigma2s))

# loop on parameters
for (idx, rho) in enumerate(rhos)
    for (idy, sigma2) in enumerate(sigma2s)

        # compute parameters
        alpha0s[idx,idy] = 1.0 / (rho * sigma2) - 1.0
        alphas[idx,idy] = (1.0 / sigma2 - 1.0) / (1.0 - rho)

    end
end

begin # first subplot

    # retrieve plot
    plt = subplots[1]

    # loop on variances
    for (idx, rho) in enumerate(rhos)
        plot!(plt, sigma2s, alpha0s[idx,:], color = 3, linestyle = mystyles[idx], primary = false)
        plot!(plt, sigma2s, alphas[idx,:], color = 2, linestyle = mystyles[idx], primary = false)
    end

    # legend lines
    for (rho, style) in zip(rhos, mystyles)
        plot!(plt, [NaN], [NaN], label = "\$\\rho = " * string(rho) * "\$", linecolor = :black, linestyle = style)
    end

    # plot attributes
    xlabel!(plt, "\$\\sigma^2\$")
    ylabel!(plt, "parameters")
    ylims!(plt, 0.0, 25.0)

end

# parameter values
sigma2s = [0.2, 0.5, 0.8]
rhos = range(0.18, 0.82, length = 100)

# initialize parameters
alphas = zeros(length(rhos), length(sigma2s))
alpha0s = zeros(length(rhos), length(sigma2s))

# loop on parameters
for (idx, rho) in enumerate(rhos)
    for (idy, sigma2) in enumerate(sigma2s)

        # compute parameters
        alpha0s[idx,idy] = 1.0 / (rho * sigma2) - 1.0
        alphas[idx,idy] = (1.0 / sigma2 - 1.0) / (1.0 - rho)

    end
end

begin # second subplot

    # retrieve plot
    plt = subplots[2]

    # loop on variances
    for (idy, sigma2) in enumerate(sigma2s)
        plot!(plt, rhos, alpha0s[:,idy], color = 3, linestyle = mystyles[idy], primary = false)
        plot!(plt, rhos, alphas[:,idy], color = 2, linestyle = mystyles[idy], primary = false)
    end

    # legend lines
    for (sigma2, style) in zip(sigma2s, mystyles)
        plot!(plt, [NaN], [NaN], label = "\$\\sigma^2 = " * string(sigma2) * "\$", linecolor = :black, linestyle = style)
    end

    # plot attributes
    plot!(plt, legend = :top)
    xlabel!(plt, "\$\\rho\$")
    # ylabel!(plt, "parameters")
    ylims!(plt, 0.0, 25.0)

end

begin

    # create legend
    plegend = plot(axis = false, grid = false, legend = :bottom, legend_column = 2, fontfamily = "Computer Modern")

    # legend patches
    plot!(plegend, [NaN], [NaN], seriestype = :bar, label = "\$\\alpha_0\$", fillcolor = 3, linewidth = 0)
    plot!(plegend, [NaN], [NaN], seriestype = :bar, label = "\$\\alpha\$", fillcolor = 2, linewidth = 0)

    # combine legend
    plot(subplots, plegend, layout = @layout([plt; leg{0.1h}]), size = (960, 420))

end 

# save figure
savefig("figures/moments_hdp.pdf")
