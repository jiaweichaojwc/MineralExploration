# test/test_realdata_csv.jl
#
# Self-contained validation of the real-data CSV workflow.
# Only requires CSV, DataFrames, Statistics, and Test (all stdlib or lightweight).
#
# This test can be run in any environment where those four packages are available:
#   julia --project=test test/test_realdata_csv.jl
#
# It validates:
#   1. data/my_obs_xy.csv format and content.
#   2. DataFrames.groupby/combine de-duplication (fully-qualified to avoid
#      GeoStats export clash).
#   3. coords/ore_quals matrix shapes match what realdata_pomcpow.jl expects.
#   4. The pmap→map fix is present in src/geostats.jl (static analysis).
#   5. No script-style code (CSV.read, main(), bare groupby) in src/geostats.jl.

using Test, CSV, DataFrames, Statistics

const REPO_ROOT = joinpath(@__DIR__, "..")
const CSV_PATH  = joinpath(REPO_ROOT, "data", "my_obs_xy.csv")
const GEO_SRC   = joinpath(REPO_ROOT, "src", "geostats.jl")

# ── Test 1 : CSV load and de-duplication ─────────────────────────────────────
@testset "CSV load and de-duplication" begin
    @test isfile(CSV_PATH)

    df = CSV.read(CSV_PATH, DataFrame)
    df[!, :i]   = Int.(df[!, :i])
    df[!, :j]   = Int.(df[!, :j])
    df[!, :ore] = Float64.(df[!, :ore])

    # Fully-qualified to avoid clash with GeoStats.groupby / GeoStats.combine
    gdf = DataFrames.combine(
        DataFrames.groupby(df, [:i, :j]),
        :ore => mean => :ore,
    )

    println("  Raw rows: $(nrow(df)), unique (i,j): $(nrow(gdf))")

    @test nrow(df)  == 28
    @test nrow(gdf) == 25

    @test all(1 .<= gdf[!, :i] .<= 50)
    @test all(1 .<= gdf[!, :j] .<= 50)
    @test all(0.0 .<= gdf[!, :ore] .<= 1.0)

    # Duplicate (23,24) → averaged
    row = filter(r -> r.i == 23 && r.j == 24, gdf)
    @test nrow(row) == 1
    @test isapprox(row[1, :ore], mean([0.8183, 0.8210]), atol=1e-4)

    # Duplicate (25,25) → averaged
    row25 = filter(r -> r.i == 25 && r.j == 25, gdf)
    @test nrow(row25) == 1
    @test isapprox(row25[1, :ore], mean([0.8444, 0.8780]), atol=1e-4)

    # Duplicate (5,5) → averaged
    row5 = filter(r -> r.i == 5 && r.j == 5, gdf)
    @test nrow(row5) == 1
    @test isapprox(row5[1, :ore], mean([0.2399, 0.2065]), atol=1e-4)

    # Mainbody cluster (i,j ∈ 22..28) should be higher-ore than periphery
    center_rows = filter(r -> r.i in 22:28 && r.j in 22:28, gdf)
    edge_rows   = filter(r -> (r.i < 10 || r.i > 40) && (r.j < 10 || r.j > 40), gdf)
    @test mean(center_rows.ore) > mean(edge_rows.ore)
    println("  Mainbody cluster mean: $(round(mean(center_rows.ore),digits=4))")
    println("  Peripheral mean:       $(round(mean(edge_rows.ore),digits=4))")

    # Matrix shapes expected by realdata_pomcpow.jl / inject logic
    coords    = permutedims(Matrix{Int64}(gdf[:, [:i, :j]]))
    ore_quals = Vector{Float64}(gdf[!, :ore])
    @test size(coords)      == (2, 25)
    @test length(ore_quals) == 25
    @test eltype(coords)    == Int64
    @test eltype(ore_quals) == Float64
end

# ── Test 2 : pmap → map fix in src/geostats.jl ───────────────────────────────
@testset "solve_nopreproc serial map fix" begin
    @test isfile(GEO_SRC)
    src = read(GEO_SRC, String)

    # Guard must be present
    @test occursin("length(procs) <= 1", src)

    # Serial map call must be present
    @test occursin("map(1:nreals(problem)", src)

    # pmap still present for multi-process opt-in
    @test occursin("pmap", src)

    # Old unconditional `reals = GeoStats...pmap(...)` must be gone
    @test !occursin(r"reals\s*=\s*GeoStats\.GeoStatsBase\.pmap"s, src)

    println("  ✓ length(procs) <= 1 guard present")
    println("  ✓ serial map path present")
    println("  ✓ pmap preserved for multi-process opt-in")
    println("  ✓ unconditional pmap removed")
end

# ── Test 3 : No script-style code in library source ──────────────────────────
@testset "No script code in src/geostats.jl" begin
    src = read(GEO_SRC, String)

    @test !occursin("CSV.read",  src)
    @test !occursin("main()",    src)
    @test !occursin("groupby",   src)   # no groupby calls (library, not script)
    @test !occursin("combine(",  src)   # no combine calls (library, not script)

    println("  ✓ CSV.read absent")
    println("  ✓ No top-level main() call")
    println("  ✓ No bare groupby/combine at top level")
end

println("\nAll real-data validation tests passed ✓")
