"""
    mutable struct CRF

"""
mutable struct CRF

    # dimensions
    n::Int64    # customers
    d::Int64    # restaurants
    k::Int64    # unique dishes

    # observations
    Xstar::Vector{Int64}        # unique values (dishes)
    counts::Matrix{Int64}       # counts per restaurant and dish

    # assignment vectors (dim n)
    restaurants::Vector{Int64}  # restaurant indexes
    dishes::Vector{Int64}       # dish indexes
    tables::Vector{Int64}       # table indexes

    # auxiliary counts
    q::Vector{Int64}    # customers per table (dim rsum)
    r::Matrix{Int64}    # tables per restaurant and dish

    # lookup vectors (dim rsum)
    table_dish::Vector{Int64}   # dish index for each table
    table_rest::Vector{Int64}   # restaurant index for each table

    # preallocated vector (dim rsum)
    masses::Vector{Float64}

end

"""
    CRF(X)

"""
function CRF(X::Vector{Vector{T}}) where T
        
    # number of groups
    d = length(X)
    
    # vector of observations
    allX = vcat(X...)
    n = length(allX)
    
    # find unique values
    Xstar = sort(unique(allX))
    k = length(Xstar)
    
    # count occurrences for each group and value
    counts = zeros(Int64, d, k)
    for (i, X_group) in enumerate(X)
        for (j, value) in enumerate(Xstar)
            counts[i,j] = sum(X_group .== value)
        end
    end

    # restaurants (dim n)
    restaurants = vcat([fill(i, length(X_group)) for (i, X_group) in enumerate(X)]...)
    
    # dishes (dim n)
    dishes = zeros(Int64, n)
    for (j, value) in enumerate(Xstar)
        dishes[allX .== value] .= j
    end

    # tables (dim n)
    tables = Vector{Int64}(undef, n)

    # customers per table (dim rsum)
    rsum = 4
    q = zeros(Int64, rsum)

    # tables per restaurant and dish
    r = zeros(Int64, d, k)

    # lookup vectors (dim rsum)
    table_dish = zeros(Int64, rsum)
    table_rest = zeros(Int64, rsum)

    # auxiliary vector
    masses = zeros(Float64, rsum + 1)
    
    return CRF(n, d, k, Xstar, counts, restaurants, dishes, tables, q, r, table_dish, table_rest, masses)

end # CRF

"""
    gibbs_step_initialize(crf, alpha, alpha0)

"""
function gibbs_step_initialize(crf::CRF, alpha::Float64, alpha0::Float64)

    for cust in range(1, crf.n)

        # compute masses for new allocation
        compute_masses(cust, crf, alpha, alpha0)

        # sample new allocation
        new_allocation(cust, crf)

    end

end # gibbs_step_initialize

"""
    gibbs_step(crf, alpha, alpha0)

"""
function gibbs_step(crf::CRF, alpha::Float64, alpha0::Float64)

    for cust in range(1, crf.n)

        # remove customer from data structure
        remove_customer(cust, crf)

        # compute masses for new allocation
        compute_masses(cust, crf, alpha, alpha0)

        # sample new allocation
        new_allocation(cust, crf)

    end

end # gibbs_step

"""
    remove_customer(cust, crf)

"""
function remove_customer(cust::Int64, crf::CRF)

    # retrieve table
    table = crf.tables[cust]

    # update counts
    crf.q[table] -= 1       # customers per table 

    if crf.q[table] == 0    # no customers at table

        # tables per restaurant and dish
        dish, rest = crf.dishes[cust], crf.restaurants[cust]
        crf.r[rest, dish] -= 1

        # update lookup vectors
        crf.table_dish[table] = -1
        crf.table_rest[table] = -1
    end

    # update indices
    crf.tables[cust] = 0

end # remove_customer

"""
    compute_masses(cust, crf, alpha, alpha0)

"""
function compute_masses(cust::Int64, crf::CRF, alpha::Float64, alpha0::Float64)

    # retrieve dish and restaurant
    dish, rest = crf.dishes[cust], crf.restaurants[cust]

    # zero masses
    crf.masses[:] .= 0.0

    # sit at old table
    # idxs = findall((crf.table_dish .== dish) .& (crf.table_rest .== rest))
    for (table, qtable) in enumerate(crf.q)
        if crf.table_dish[table] == dish && crf.table_rest[table] == rest
            crf.masses[table] = qtable 
        end
    end
    
    # sit at new table
    crf.masses[end] = alpha * sum(crf.r[:, dish]) / (alpha0 + sum(crf.r))

end # compute_masses

"""
    new_allocation(cust, crf)

"""
function new_allocation(cust::Int64, crf::CRF)

    # retrieve masses
    masses = crf.masses
    
    # sample table index
    if sum(masses) == 0.0
        table = length(crf.q) + 1
    else
        table = sample_categorical(masses)
    end

    if table == length(crf.q) + 1   # sit at new table

        # find new table label
        table = find_table(crf)

        # tables per restaurant and dish
        dish, rest = crf.dishes[cust], crf.restaurants[cust]
        crf.r[rest, dish] += 1

        # update lookup vectors
        crf.table_dish[table] = dish
        crf.table_rest[table] = rest

    end

    # update counts
    crf.q[table] += 1       # customers per table 
    
    # update indices
    crf.tables[cust] = table

end # new_allocation

"""
    find_table(crf)

"""
function find_table(crf::CRF)

    # find empty table
    table = findfirst(crf.q .== 0)
    if !isnothing(table)
        return table
    end
        
    # retrieve dimension
    rsum = length(crf.q)
    
    # customers per table
    append!(crf.q, zeros(Int64, rsum))

    # lookup vectors
    append!(crf.table_dish, zeros(Int64, rsum))
    append!(crf.table_rest, zeros(Int64, rsum))

    # auxiliary vector
    append!(crf.masses, zeros(Float64, rsum))

    return rsum + 1

end # find_table
