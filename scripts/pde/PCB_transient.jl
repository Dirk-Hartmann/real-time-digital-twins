# Transient 2D heat conduction on a PCB geometry with the finite element method.
#
# Integrates the semi-discrete FE system  M ⋅ dT/dt = -K ⋅ T + s_c ⋅ c(t) + s_h ⋅ h(t)
# over a 10 s window, starting from a uniform temperature equal to the Dirichlet
# boundary value (T = 0). The scalar cooling / heating drives c(t) and h(t) are
# random input signals from an Ornstein-Uhlenbeck process passed through a
# lowpass filter. Dirichlet T = 0 is held on the left and right edges (top and
# bottom stay zero-Neumann).
# The `pcb_thermal_model` is marched with its implicit-Euler `predict` map.
# The evolving temperature field is written as
# an animated GIF.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using SparseArrays, LinearAlgebra, Plots, Images, Random, BSON

println("\nSimulating and visualising the transient PCB heat conduction problem...\n")

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

# --- Time grid over the 10 s simulation window ---
δt = 0.10; t_end = 10.0; ts = 0.0:δt:t_end

# --- Random input signals c(t), h(t): Ornstein-Uhlenbeck + lowpass, as callables ---
rng = MersenneTwister(1)
c_of = signal_random(ts; θ = 0.3, σ = 2.0, τ = 0.5, lo = -2.0, hi = 0.0, rng = rng)
h_of = signal_random(ts; θ = 0.3, σ = 2.0, τ = 0.5, lo =  0.0, hi = 2.0, rng = rng)
c, h = c_of.(ts), h_of.(ts)                          # samples for plotting / storage

# --- Assemble the finite element model (Dirichlet T = 0 on the left and right edges) ---
model = pcb_thermal_model(g, field_κ, field_s_c, field_s_h; bc = (left = 0.0, right = 0.0))

# --- Integrate  M ⋅ dT/dt = -K ⋅ T + s_c ⋅ c(t) + s_h ⋅ h(t)  with implicit Euler ---
T = [zeros(g.nx * g.ny) for _ in eachindex(ts)]
for i in 2:length(ts)
    T[i] = model.predict(model.params, T[i-1], (c[i], h[i]), δt)
    i % 100 == 0 && println("  Time-step $i / $(length(ts))")
end



# ============================== VISUALIZATION ===============================

# --- Reshape nodal vectors to (ny x nx) grids ---
frames = [tomat(g, t) for t in T]
Tstack = cat(frames...; dims = 3)                    # (ny x nx x nsteps)

# --- Animate the temperature evolution with fixed color limits ---
clims = extrema(Tstack)
anim = @animate for k in eachindex(ts)
    k % 100 == 0 && println("  frame $k / $(length(ts))")
    heatmap(frames[k]; title = "T at t = $(round(ts[k], digits = 2)) s  " *
            "(c = $(round(c[k], digits = 2)), h = $(round(h[k], digits = 2)))",
            color = :thermal, clims = clims, yflip = true, aspect_ratio = 1,
            axis = false, ticks = false, size = (600, 560))
end
resultdir = joinpath(ROOT, "results", "pde"); mkpath(resultdir)
gifpath = joinpath(resultdir, "PCB_transient.gif")
gif(anim, gifpath, fps = 20)
