mutable struct LUParams
    C₂₂::Matrix{Float64}
    A₂₁::Matrix{Float64}
    L₁₁::LowerTriangular{Float64, Matrix{Float64}}
    L₂₂::LowerTriangular{Float64, Matrix{Float64}}
    dlocs::Vector{Int64}
    slocs::Vector{Int64}
    lugs::LUGS
end

function LUParams(γ::Variogram, domain::CartesianGrid)
    z₁ = Float64[0.0]
    d₂ = Float64[0.0]
    slocs = [l for l in 1:nelements(domain)] # if l ∉ dlocs]
    dlocs = Int64[]
    𝒟s = [centroid(domain, i) for i in slocs]
    C₂₂ = sill(γ) .- GeoStats.pairwise(γ, 𝒟s)
    A₂₁ = zeros(Float64, 0, 0)
    L₁₁ = cholesky(A₂₁).L
    K = Symmetric(C₂₂)
    K += γ.nugget.*Matrix(I, size(K))
    L₂₂ = cholesky(K).L
    lugs = LUGS(:ore => (mean=0.0, variogram=γ,))
    return LUParams(C₂₂, A₂₁, L₁₁, L₂₂, dlocs, slocs, lugs)
end

@with_kw struct GeoStatsDistribution <: GeoDist
    grid_dims::Tuple{Int64, Int64, Int64} = (50, 50, 1)
    data::RockObservations = RockObservations()
    domain::CartesianGrid{2, Int64} = CartesianGrid{Int64}(50, 50)
    mean::Float64 = 0.3
    variogram::Variogram = SphericalVariogram(sill=0.005, range=30.0,
                                            nugget=0.0001)
    lu_params::LUParams = LUParams(variogram, domain)
end

function update!(d::GeoStatsDistribution, o::RockObservations)
    d.data.ore_quals = o.ore_quals
    d.data.coordinates = o.coordinates

    table = DataFrame(ore=d.data.ore_quals .- d.mean)
    domain = PointSet(d.data.coordinates)
    pdata = georef(table, domain)
    pdomain = d.domain

    var = :ore
    varparams = d.lu_params.lugs.vparams[:ore]
    vmapping = map(pdata, pdomain, (var,), varparams.mapping)[var]
    dlocs = Int[]
    for (loc, dloc) in vmapping
        push!(dlocs, loc)
    end
    d.lu_params.dlocs = dlocs

    𝒟d = [centroid(pdomain, i) for i in dlocs]
    𝒟s = [centroid(pdomain, i) for i in d.lu_params.slocs]

    γ = d.variogram
    C₁₁ = sill(γ) .- GeoStats.pairwise(γ, 𝒟d)
    C₁₂ = sill(γ) .- GeoStats.pairwise(γ, 𝒟d, 𝒟s)
    L₁₁ = varparams.factorization(Symmetric(C₁₁)).L
    B₁₂ = L₁₁ \ C₁₂
    A₂₁ = B₁₂'

    K = Symmetric(d.lu_params.C₂₂ - A₂₁*B₁₂)
    K += d.variogram.nugget.*Matrix(I, size(K))
    L₂₂ = varparams.factorization(K).L

    d.lu_params.A₂₁ = A₂₁
    d.lu_params.L₁₁ = L₁₁
    d.lu_params.L₂₂ = L₂₂
end

function calc_covs(d::GeoStatsDistribution, problem)
    pdata = data(problem)
    pdomain = domain(problem)

    var = :ore
    varparams = d.lu_params.lugs.vparams[:ore]
    if hasdata(problem)
        vmapping = map(pdata, pdomain, (var,), varparams.mapping)[var]
    else
        vmapping = Dict()
    end
    z₁ = Float64[]
    for (loc, dloc) in vmapping
        push!(z₁, pdata[var][dloc])
    end

    𝒟d = [centroid(pdomain, i) for i in d.lu_params.dlocs]
    𝒟s = [centroid(pdomain, i) for i in d.lu_params.slocs]

    fact = varparams.factorization
    if isempty(d.lu_params.dlocs)
        d₂  = zero(Float64)
    else
        B₁₂ = d.lu_params.A₂₁'
        d₂ = d.lu_params.A₂₁ * (d.lu_params.L₁₁ \ z₁)
    end
    return (d₂, z₁)
end

"""
    solve(problem, solver; procs=[myid()])
Solve the simulation `problem` with the simulation `solver`,
optionally using multiple processes `procs`.
### Notes
When only one process is provided (the default), `solvesingle` is called
serially with `map` to avoid `DimensionMismatch` errors from Distributed
on Windows / Julia 1.6.  `pmap` is only used when multiple worker processes
are explicitly supplied.
"""
function solve_nopreproc(problem::SimulationProblem, solver::LUGS, preproc::Dict; procs=[GeoStats.GeoStatsBase.myid()])
    # sanity checks
    @assert targets(solver) ⊆ name.(variables(problem)) "invalid variables in solver"

    # dictionary with variable types
    mactypeof = Dict(name(v) => mactype(v) for v in variables(problem))

    # # optional preprocessing
    # preproc = preprocess(problem, solver)

    # pool of worker processes
    pool = GeoStats.GeoStatsBase.CachingPool(procs)

    # list of covariables
    allcovars = covariables(problem, solver)
    # simulation loop
    results = []

    for covars in allcovars
        # simulate covariables — use serial map when running on a single process
        # to avoid DimensionMismatch errors from Distributed on Windows/Julia 1.6
        reals = if length(procs) <= 1
            map(1:nreals(problem)) do _
                solvesingle(problem, covars, solver, preproc)
            end
        else
            GeoStats.GeoStatsBase.pmap(pool, 1:nreals(problem)) do _
                solvesingle(problem, covars, solver, preproc)
            end
        end

        # rearrange realizations
        vnames = covars.names
        vtypes = [mactypeof[var] for var in vnames]
        vvects = [Vector{V}[] for V in vtypes]
        rtuple = (; zip(vnames, vvects)...)
        for real in reals
            for var in vnames
                push!(rtuple[var], real[var])
            end
        end
        push!(results, rtuple)
    end

    # merge results into a single dictionary
    pdomain = domain(problem)
    preals  = reduce(merge, results)

    Ensemble(pdomain, preals)
end

function Base.rand(rng::AbstractRNG, d::GeoStatsDistribution, n::Int64=1)
    if isempty(d.data.coordinates) # Unconditional simulation
        problem = SimulationProblem(d.domain, (:ore => Float64), n)
    else
        table = DataFrame(ore=d.data.ore_quals .- d.mean)
        domain = PointSet(d.data.coordinates)
        geodata = georef(table, domain)
        problem = SimulationProblem(geodata, d.domain, (:ore), n)
    end
    conames = (:ore,)
    d₂, z₁ = calc_covs(d, problem)
    μ = 0.0
    coparams = [(z₁, d₂, d.lu_params.L₂₂, μ, d.lu_params.dlocs, d.lu_params.slocs),]
    preproc = Dict()
    push!(preproc, conames => coparams)
    solution = solve_nopreproc(problem, d.lu_params.lugs, preproc)
    ore_maps = Array{Float64, 3}[]
    for s in solution[:ore]
        ore_2D = reshape(s[1:prod(d.grid_dims)], d.grid_dims) .+ d.mean
        ore_map = repeat(ore_2D, outer=(1, 1, 1))
        push!(ore_maps, ore_map)
    end
    if n == 1
        return ore_maps[1]
    else
        return ore_maps
    end
end

Base.rand(d::GeoStatsDistribution, n::Int64=1) = Base.rand(Random.GLOBAL_RNG, d, n)

# function solve_gp(d::GeoStatsDistribution)
#     table = DataFrame(porosity=d.data.ore_quals)
#     domain = PointSet(d.data.coordinates)
#     geodata = georef(table, domain)
#     problem = EstimationProblem(geodata, d.domain, :ore)
#     solver = Kriging(
#                         :ore => ( mean=d.mean,
#                                     variogram=d.variogram
#                                        )
#                          )
#     solution = GeoStats.solve(problem, solver)
#     ore_mean = reshape(solution[:ore], d.grid_dims)
#     ore_var = reshape(solution[:ore_variance], d.grid_dims)
#     return (ore_mean, ore_var)
# end
