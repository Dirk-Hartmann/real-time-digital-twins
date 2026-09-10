# Visualise the Lorenz system over time. The model is marched with its linearly-implicit
# trapezoidal `predict` map, starting from a point near the attractor. The resulting trajectory
# is shown as the 3D butterfly attractor together with the time series of its three
# components, and an animated GIF traces the state as it wanders between the two lobes.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Plots

println("\nSimulating a single Lorenz system trajectory...\n")

# --- Include repo files relative to the root (avoids repeated joinpath calls) ---
const ROOT = normpath(@__DIR__, "..", "..")
use(parts...) = Base.include(Main, joinpath(ROOT, parts...))
use("models", "ode", "ode.jl")

# --- Time grid and constant (zero) forcing ---
δt = 0.01; t_end = 40.0; ts = 0.0:δt:t_end
u = zeros(3)

# --- Simulate the Lorenz model over the specified time horizon from a point near the attractor---
model = lorenz_model()
X = Matrix{Float64}(undef, 3, length(ts)); X[:, 1] = [1.0, 1.0, 1.0]
for i in 2:length(ts)
    X[:, i] = model.predict(model.params, X[:, i-1], u, δt)
    i % 100 == 0 && println("  Time-step $i / $(length(ts))")
end
x1, x2, x3 = X[1, :], X[2, :], X[3, :]



# ============================== VISUALIZATION ===============================

# --- Static figure: 2D x₁–x₃ phase projection and the three component time series ---
attractor = plot(x1, x3; xlabel = "x₁", ylabel = "x₃",
                 title = "Lorenz attractor: phase portrait (x₁–x₃)", legend = false, lw = 1,
                 lc = :viridis, line_z = ts, colorbar = false)
series = plot(ts, [x1 x2 x3]; xlabel = "t", title = "components over time",
              label = ["x₁(t)" "x₂(t)" "x₃(t)"], lw = 1)
plt = plot(attractor, series, layout = (1, 2), size = (1200, 500))

# --- Animated trajectory tracing the state through the x₁–x₃ phase plane ---
step = 20                                            # frames every `step` time steps
frame_ids = 2:step:length(ts)
anim = @animate for (frame, k) in enumerate(frame_ids)
    frame % 100 == 0 && println("  frame $frame / $(length(frame_ids))")
    plot(x1[1:k], x3[1:k]; xlabel = "x₁", ylabel = "x₃",
         title = "Lorenz attractor, t = $(round(ts[k], digits = 1))",
         legend = false, lw = 1, lc = :viridis, line_z = ts[1:k], colorbar = false,
         xlims = extrema(x1), ylims = extrema(x3), size = (700, 600))
    scatter!([x1[k]], [x3[k]]; mc = :red, ms = 4)
end

# --- Store the figure and the animation in results/ode ---
resultdir = joinpath(ROOT, "results", "ode"); mkpath(resultdir)
savefig(plt, joinpath(resultdir, "Lorenz_trajectory.png"))
gif(anim, joinpath(resultdir, "Lorenz_trajectory.gif"), fps = 25)
