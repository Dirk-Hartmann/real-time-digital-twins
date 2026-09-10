# Generate a training dataset of Lorenz trajectories. One hundred trajectories are
# rolled forward with the linearly-implicit trapezoidal `predict` map over the window [0, 10] from
# random initial conditions scattered around the attractor.
# The trajectories are stacked into a single array and persisted to a BSON file in
# data/, and their x–z phase projections are overlaid in a static figure.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Random, BSON, Plots

println("\nGenerating an ensemble of Lorenz training trajectories for data-driven algorithms...\n")

# --- Include repo files relative to the root (avoids repeated joinpath calls) ---
const ROOT = normpath(@__DIR__, "..", "..")
use(parts...) = Base.include(Main, joinpath(ROOT, parts...))
use("models", "ode", "ode.jl")

# --- Time grid, forcing, and ensemble size ---
δt = 0.01; t_end = 10.0; ts = 0.0:δt:t_end
u = zeros(3)
n_traj = 100

# --- Random initial conditions near the attractor ---
rng = MersenneTwister(1)
X0 = [10.0 .* randn(rng, 3) for _ in 1:n_traj]

# --- Simulate every trajectory forward with the linearly-implicit trapezoidal model ---
model = lorenz_model()
X = Array{Float64}(undef, 3, length(ts), n_traj)     # (state x time x trajectory)
for j in 1:n_traj
    X[:, 1, j] = X0[j]
    for i in 2:length(ts)
        X[:, i, j] = model.predict(model.params, X[:, i-1, j], u, δt)
    end
    j % 10 == 0 && println("  Trajectory $j / $n_traj")
end

# --- Store the dataset in data/ and the figure in results/ode ---
BSON.@save joinpath(ROOT, "data", "Lorenz_training.bson") t=collect(ts) X=X



# ============================== VISUALIZATION ===============================

# --- Static figure: overlay of all x–z phase projections ---
plt = plot(; xlabel = "x", ylabel = "z", legend = false,
           title = "$n_traj Lorenz attractor: x₁–x₃ projection",
           size = (800, 700))
for j in 1:n_traj
    plot!(plt, X[1, :, j], X[3, :, j]; lw = 0.5, lc = :viridis,
          line_z = ts, alpha = 0.5, colorbar = false)
end

# --- Store the figure and the animation in results/ode ---
resultdir = joinpath(ROOT, "results", "ode"); mkpath(resultdir)
savefig(plt, joinpath(resultdir, "Lorenz_training_data.png"))
