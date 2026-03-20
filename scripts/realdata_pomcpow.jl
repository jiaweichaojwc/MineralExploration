# scripts/realdata_pomcpow.jl
#
# Run POMCPOW planning on real drill-hole data loaded from a CSV file.
#
# Expected CSV format (data/my_obs_xy.csv):
#   i,j,ore
#   3,7,0.42
#   ...
#
# Columns:
#   i, j  — integer grid coordinates (1-based)
#   ore   — observed ore quality (Float64)
#
# Duplicate (i,j) pairs are aggregated by taking the mean ore value.
# DataFrames functions are fully qualified to avoid name-clash warnings
# between GeoStats and DataFrames exports (groupby / combine).
#
# Usage:
#   julia --project=. scripts/realdata_pomcpow.jl
#
# Note: On Windows / Julia 1.6 the default serial path in solve_nopreproc
# is used automatically (pmap is only invoked when multiple worker processes
# are explicitly provided via the `procs` keyword).

using POMDPs
using POMCPOW
using POMDPSimulators
using CSV
using DataFrames
using Statistics
using Random

using MineralExploration

function main()
    rng = MersenneTwister(42)

    # ── 1) Load and de-duplicate observations ────────────────────────────────
    csv_path = joinpath(@__DIR__, "..", "data", "my_obs_xy.csv")
    df = CSV.read(csv_path, DataFrame)
    df[!, :i]   = Int.(df[!, :i])
    df[!, :j]   = Int.(df[!, :j])
    df[!, :ore] = Float64.(df[!, :ore])

    # Use fully-qualified DataFrames functions to avoid GeoStats export clash
    gdf = DataFrames.combine(
        DataFrames.groupby(df, [:i, :j]),
        :ore => mean => :ore
    )

    coords    = permutedims(Matrix{Int64}(gdf[:, [:i, :j]]))  # 2 × N
    ore_quals = Vector{Float64}(gdf[!, :ore])

    println("Raw rows: ",        nrow(df))
    println("After unique (i,j): ", nrow(gdf))

    # ── 2) Build POMDP and inject initial observations ────────────────────────
    # Allow at least 10 additional bores on top of the pre-loaded observations
    # so the planner has room to drill new holes during the episode.
    EXTRA_BORES  = 10
    MAX_BORES    = max(200, length(ore_quals) + EXTRA_BORES)
    GRID_SPACING = 1
    DELTA        = 2

    m = MineralExplorationPOMDP(
        max_bores    = MAX_BORES,
        delta        = DELTA,
        grid_spacing = GRID_SPACING,
        mainbody_gen = SingleFixedNode(),
        max_movement = 0,
    )

    append!(m.initial_data.ore_quals, ore_quals)
    m.initial_data.coordinates = hcat(m.initial_data.coordinates, coords)
    println("Initial obs count = ", length(m.initial_data.ore_quals))

    # ── 3) Initial state distribution and belief ──────────────────────────────
    ds0 = POMDPs.initialstate(m)

    # MEBeliefUpdater uses GeoStatsDistribution internally.
    # solve_nopreproc now falls back to serial `map` when only one process is
    # available, which avoids the DimensionMismatch crash on Windows/Julia 1.6.
    up = MEBeliefUpdater(m, 500, 2.0)

    println("Initializing belief (GeoStats conditional simulation)...")
    b0 = POMDPs.initialize_belief(up, ds0)
    println("Belief initialized with $(length(b0.particles)) particles.")

    # ── 4) POMCPOW planner ────────────────────────────────────────────────────
    next_action = NextActionSampler()
    solver = POMCPOWSolver(
        tree_queries       = 500,
        check_repeat_obs   = true,
        check_repeat_act   = true,
        next_action        = next_action,
        k_action           = 2.0,
        alpha_action       = 0.25,
        k_observation      = 2.0,
        alpha_observation  = 0.1,
        criterion          = POMCPOW.MaxUCB(100.0),
        final_criterion    = POMCPOW.MaxQ(),
        estimate_value     = 0.0,
    )
    planner = POMDPs.solve(solver, m)

    # ── 5) Sample a true initial state ────────────────────────────────────────
    s0 = rand(rng, ds0)

    # ── 6) Run one episode via stepthrough ────────────────────────────────────
    println("Starting POMCPOW simulation (max $(MAX_BORES) steps)...")
    total_reward = 0.0
    t = 0
    for (s, a, r, bp) in stepthrough(m, planner, up, b0, s0,
                                      "s,a,r,bp", max_steps=MAX_BORES+5)
        t += 1
        total_reward += POMDPs.discount(m)^(t - 1) * r
        println("t=$t  a=$(a.type)  coords=$(a.coords)  r=$r  total_r=$(round(total_reward, digits=4))")
        if s.decided
            println("Decision made at t=$t: $(a.type)")
            break
        end
    end
    println("Done.  Discounted return = $(round(total_reward, digits=4))")
end

main()
