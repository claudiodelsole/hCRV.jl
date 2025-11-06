# load modules
using hCRV

# imports
import SpecialFunctions: gamma, eulergamma
using Plots

# functions
function tail_integral(x::Float64)::Float64
    
    if log(x) <= -256.0     # asymptotic behaviour
        return - eulergamma - log(x)
    end
    
    return gamma(0.0, x)

end # tail_integral

function tail_integral_derivative(x::Float64)::Float64

    if log(x) <= -256.0     # asymptotic behaviour
        return - 1.0 / x
    end
    
    return - exp(-x) / x

end # tail_integral_derivative

# solution value
value = 2.0

# starting point
startr = 0.25

# first iteration
stopr = startr - (tail_integral(startr) - value) / tail_integral_derivative(startr)

# starting point
startl = 0.025

# iterations
stops = zeros(4)
stops[1] = startl - (tail_integral(startl) - value) / tail_integral_derivative(startl)

for iter in range(2,4)
    stops[iter] = stops[iter-1] - (tail_integral(stops[iter-1]) - value) / tail_integral_derivative(stops[iter-1])
end

# solution
sol = hCRV.newton(value, tail_integral, tail_integral_derivative, startl)

# initialize plot
subplots = plot(layout = (1, 2), legend = false, size = (960, 360), fontfamily = "Computer Modern")

begin # first subplot

    # retrieve plot
    plt = subplots[1]

    # plot exponential integral
    evalpoints = range(0.01, 0.4, length = 1000)
    plot!(plt, evalpoints, [tail_integral(x) for x in evalpoints], color = 1)

    # plot solution
    hline!(plt, [value], linecolor = 5, linestyle = :dashdot)
    plot!(plt, [sol, sol], [0.5, value], linecolor = 1, linestyle = :dash)

    # plot newton algorithm
    plot!(plt, [startr, startr], [0.5, tail_integral(startr)], linecolor = 2, linestyle = :dash)
    plot!(plt, [startr, stopr], [tail_integral(startr), value], linecolor = 2)

    # plot newton algorithm
    plot!(plt, [startl, startl], [0.5, tail_integral(startl)], linecolor = 3, linestyle = :dash)
    plot!(plt, [startl, stops[1]], [tail_integral(startl), value], linecolor = 3)
    for (iter, stop) in enumerate(stops[1:end-1])
        plot!(plt, [stop, stop], [value, tail_integral(stop)], linecolor = 3, linestyle = :dash)
        plot!(plt, [stop, stops[iter+1]], [tail_integral(stop), value], linecolor = 3)
    end

    # plot attributes
    xlims!(plt, -0.1, 0.35)
    ylims!(plt, 0.7, 3.5)

end

begin # second subplot

    # retrieve plot
    plt = subplots[2]

    # plot exponential integral
    evalpoints = range(0.01, 0.1, length = 1000)
    plot!(plt, evalpoints, [tail_integral(x) for x in evalpoints], color = 1)

    # plot solution
    hline!(plt, [value], linecolor = 5, linestyle = :dashdot)
    plot!(plt, [sol, sol], [0.5, value], linecolor = 1, linestyle = :dash)

    # plot newton algorithm
    plot!(plt, [startl, startl], [0.5, tail_integral(startl)], linecolor = 3, linestyle = :dash)
    plot!(plt, [startl, stops[1]], [tail_integral(startl), value], linecolor = 3)
    for (iter, stop) in enumerate(stops[1:end-1])
        plot!(plt, [stop, stop], [value, tail_integral(stop)], linecolor = 3, linestyle = :dash)
        plot!(plt, [stop, stops[iter+1]], [tail_integral(stop), value], linecolor = 3)
    end

    # plot attributes
    xlims!(plt, 0.0, 0.1)
    ylims!(plt, 1.6, 3.2)

end

# save figure
savefig("figures/newton.pdf")
