# Generate a training dataset of PCB transient temperature fields. Ten transients are
# rolled forward with the implicit-Euler `predict` map of the `pcb_thermal_model` over the
# 10 s window, each driven by a fresh pair of random cooling / heating signals c(t), h(t)
# (Ornstein-Uhlenbeck process + lowpass filter). The geometry, grid and discretisation
# follow PCB_transient.jl; the ensemble is stacked into single arrays and persisted to a
# BSON file in data/, and the drives are overlaid in a static figure.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using SparseArrays, LinearAlgebra, Random, BSON, Plots, Images

println("\nGenerating an ensemble of PCB transient temperature trajectories for data-driven algorithms...\n")

# --- Include repo files relative to the root (avoids repeated joinpath calls) ---
const ROOT = normpath(@__DIR__, "..", "..")
use(parts...) = Base.include(Main, joinpath(ROOT, parts...))
use("util", "util.jl")
use("models", "pde", "pde.jl")

# --- Load geometry fields from PNGs ---
datadir = joinpath(ROOT, "models", "data")
field_κ   = Float64.(Gray.(load(joinpath(datadir, "PCB_geometry_k.png"))))
field_s_c = Float64.(Gray.(load(joinpath(datadir, "PCB_geometry_c.png"))))
field_s_h = Float64.(Gray.(load(joinpath(datadir, "PCB_geometry_h.png"))))
nely, nelx = size(field_κ)
g = Grid(nelx, nely)

# --- Time grid over the 10 s window and ensemble size ---
δt = 0.10; t_end = 10.0; ts = 0.0:δt:t_end
n_sim = 10

# --- Assemble the finite element model once (Dirichlet T = 0 on the left and right edges) ---
model = pcb_thermal_model(g, field_κ, field_s_c, field_s_h; bc = (left = 0.0, right = 0.0))

# --- Simulate n_sim transients, each with fresh random cooling / heating drives ---
rng = MersenneTwister(1)
nnodes = g.nx * g.ny
T = Array{Float64}(undef, nnodes, length(ts), n_sim)     # (node x time x simulation)
C = Array{Float64}(undef, length(ts), n_sim)             # cooling drives
H = Array{Float64}(undef, length(ts), n_sim)             # heating drives
for j in 1:n_sim
    c = signal_random(ts; θ = 0.3, σ = 2.0, τ = 0.5, lo = -2.0, hi = 0.0, rng = rng).(ts)
    h = signal_random(ts; θ = 0.3, σ = 2.0, τ = 0.5, lo =  0.0, hi = 2.0, rng = rng).(ts)
    C[:, j], H[:, j] = c, h
    T[:, 1, j] = zeros(nnodes)
    for i in 2:length(ts)
        T[:, i, j] = model.predict(model.params, T[:, i-1, j], (c[i], h[i]), δt)
    end
    println("  Simulation $j / $n_sim")
end

# --- Stack cooling / heating drives into one 2D control u (input x time x simulation) ---
u = permutedims(cat(C, H; dims = 3), (3, 1, 2))

# --- Store the dataset in data/ ---
BSON.@save joinpath(ROOT, "data", "PCB_training.bson") t=collect(ts) T=T u=u nelx=g.nelx nely=g.nely δx=g.δx 



# ============================== VISUALIZATION ===============================

# --- Static figure: cooling and heating drives across the ensemble ---
pc = plot(ts, C; xlabel = "t [s]", ylabel = "c(t)", legend = false,
          title = "Cooling drives", lc = :blue, alpha = 0.5)
ph = plot(ts, H; xlabel = "t [s]", ylabel = "h(t)", legend = false,
          title = "Heating drives", lc = :red, alpha = 0.5)
plt = plot(pc, ph; layout = (2, 1),
           plot_title = "$n_sim PCB training transients (input drives)", size = (800, 600))

# --- Store the figure in results/pde ---
resultdir = joinpath(ROOT, "results", "pde"); mkpath(resultdir)
savefig(plt, joinpath(resultdir, "PCB_training_data.png"))
