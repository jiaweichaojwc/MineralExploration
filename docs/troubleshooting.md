# Troubleshooting

## Windows / Julia 1.6 — `DimensionMismatch` during GeoStats simulation

**Symptom**

```
DimensionMismatch("tried to assign ... elements to 0 destinations")
```

when calling `rand(ds0)` or `initialize_belief(up, ds0)`.

**Cause**

The original `solve_nopreproc` in `src/geostats.jl` always used Distributed
`pmap` to parallelize realizations.  On Windows with a single Julia process
(the default), `pmap` over an empty worker pool triggers the error above.

**Fix (already applied)**

`solve_nopreproc` now checks the number of available worker processes before
choosing a parallelism strategy:

- **Single process (default):** serial `map` — safe on all platforms.
- **Multiple explicit worker processes:** `pmap` with a `CachingPool` — same
  parallel behaviour as before.

No user action is required; the fix is transparent.  If you want to restore
parallel simulation, add workers before calling any planning code:

```julia
using Distributed
addprocs(4)
@everywhere using MineralExploration
# ... then run your script normally
```

---

## Precompile failures — `UndefVarError: groupby not defined`

**Symptom**

```
WARNING: both GeoStats and DataFrames export "groupby"; uses of it in module
         MineralExploration must be qualified.
UndefVarError: groupby not defined
```

**Cause**

Script-style code (`load_obs`, `main()`, bare `groupby`/`combine` calls, etc.)
was accidentally placed inside `src/geostats.jl`.  Julia executes every
top-level expression in a source file at module-load time, so any unqualified
call to an ambiguous name fails during precompilation.

**Fix**

Keep `src/` files as **library code only** — no top-level runnable scripts,
no bare `CSV.read`, `groupby`, `combine`, or `main()` calls.

For real-data exploration workflows use the dedicated script under `scripts/`:

```bash
julia --project=. scripts/realdata_pomcpow.jl
```

That script uses fully-qualified `DataFrames.groupby` / `DataFrames.combine`
to sidestep the export conflict entirely.

---

## Do not put runnable scripts inside `src/`

| Where | What |
|-------|------|
| `src/` | Library code — structs, methods, pure functions. No I/O at the top level. |
| `scripts/` | Runnable experiments — `CSV.read`, `main()`, plots, etc. |
| `test/` | Unit and integration tests via `Pkg.test()`. |

If you need to quickly prototype with real data, copy `scripts/realdata_pomcpow.jl`
and edit it — never edit files inside `src/`.
