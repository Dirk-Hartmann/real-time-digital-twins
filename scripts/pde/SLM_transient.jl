# Transient selective laser melting (SLM) thermal simulation with the finite
# element method.
#
# Integrates the semi-discrete FE system  M ⋅ dT/dt = -(K + β M) ⋅ T + q(t)  over a
# scan window, starting from a uniform temperature T = 0. A Gaussian laser spot
# of fixed power and width travels across the domain at constant velocity; at
# each step its cell-wise field is re-assembled into the nodal load q(t). Linear
# dissipation β M models heat loss to an ambient temperature of 0, and every edge
# stays zero-Neumann (no flux). The `slm_thermal_model` is marched with its
# implicit-Euler `predict` map; the moving Gaussian load q(t) is re-assembled
# from the spot centre supplied per step. The evolving field, with the laser
# position marked, is written as an animated GIF. The simulation also generates
# a single-trajectory training dataset persisted to data/SLM_training.bson for
# use in data-driven algorithms.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using SparseArrays, LinearAlgebra, Plots, BSON

println("\nSimulating and visualising a transient SLM thermal problem with a moving laser...\n")

# --- Axis tick labels rounded to 3 significant digits on all plots ---
default(formatter = y -> y isa Number ? string(round(y, sigdigits = 3)) : string(y))

# --- Include repo files relative to the root (avoids repeated joinpath calls) ---
const ROOT = normpath(@__DIR__, "..", "..")
use(parts...) = Base.include(Main, joinpath(ROOT, parts...))
use("util", "util.jl")
use("models", "pde", "pde.jl")

# --- Structured mesh and constant conductivity on the unit domain ---
nelx = nely = 200
g = Grid(nelx, nely)
field_κ = fill(1.0, nely, nelx)

# --- Laser and dissipation parameters ---
power = 1.0                                          # integrated laser power
width = 0.03                                         # Gaussian spot radius
β     = 10.0                                         # linear dissipation rate (ambient T = 0)
x0, y0 = 0.10, 0.5                                   # initial spot centre
vx, vy = 0.9, 0.0                                    # laser velocity
laser(t) = (x0 + vx * t, y0 + vy * t)                # moving spot centre

# --- Time grid over the scan window ---
δt = 0.005; t_end = 1.0; ts = 0.0:δt:t_end

# --- Assemble the finite element model (dissipation β folded into the operator) ---
model = slm_thermal_model(g, field_κ; power = power, width = width, β = β)

# --- Integrate  M ⋅ dT/dt = -(K + β M) ⋅ T + q(t)  with implicit Euler ---
nnodes = g.nx * g.ny
T = [zeros(nnodes) for _ in eachindex(ts)]
for i in 2:length(ts)
    T[i] = model.predict(model.params, T[i-1], laser(ts[i]), δt)
    i % 50 == 0 && println("  Time-step $i / $(length(ts))")
end

# --- Convert list to array and prepare training data (single trajectory) ---
T_array = cat(T...; dims = 2)                        # (nnodes × time)
x_laser = [laser(t)[1] for t in ts]
y_laser = [laser(t)[2] for t in ts]
u = [x_laser'; y_laser']                             # (2 × time)

# --- Persist the single-trajectory training dataset to data/ ---
BSON.@save joinpath(ROOT, "data", "SLM_training.bson") t=collect(ts) T=reshape(T_array, nnodes, length(ts), 1) u=reshape(u, 2, length(ts), 1) nelx=g.nelx nely=g.nely δx=g.δx
println("Saved training dataset to data/SLM_training.bson")



# ============================== VISUALIZATION ===============================

# --- Reshape nodal vectors to (ny x nx) grids ---
frames = [tomat(g, t) for t in T]
Tstack = cat(frames...; dims = 3)                    # (ny x nx x nsteps)

# --- Animate the temperature evolution with the laser spot marked ---
xs = [(ix - 1) * g.δx for ix in 1:g.nx]
ys = [(iy - 1) * g.δx for iy in 1:g.ny]
clims = extrema(Tstack)
anim = @animate for k in eachindex(ts)
    k % 50 == 0 && println("  frame $k / $(length(ts))")
    xc, yc = laser(ts[k])
    heatmap(xs, ys, frames[k]; title = "T at t = $(round(ts[k], digits = 3)) s",
            color = :thermal, clims = clims, yflip = true, aspect_ratio = 1,
            axis = false, ticks = false, size = (600, 560))
    scatter!([xc], [yc]; marker = :circle, markersize = 6, markerstrokewidth = 2,
             color = :cyan, label = false)
end
resultdir = joinpath(ROOT, "results", "pde"); mkpath(resultdir)
gifpath = joinpath(resultdir, "SLM_transient.gif")
gif(anim, gifpath, fps = 20)
