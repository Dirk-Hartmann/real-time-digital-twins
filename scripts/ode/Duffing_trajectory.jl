# Visualise the forced Duffing oscillator over time. The model is marched with its
# linearly-implicit trapezoidal `predict` map under the harmonic
# drive u = [0, γ cos(ω t)] supplied per step. For the double-well parameters (α < 0,
# β > 0) with light damping the response is chaotic; the script shows the (x₁, x₂) phase
# portrait and the position/drive time series, and an animated GIF traces the strange
# attractor in the phase plane.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Plots

println("\nSimulating a single forced Duffing oscillator trajectory...\n")

# --- Include repo files relative to the root (avoids repeated joinpath calls) ---
const ROOT = normpath(@__DIR__, "..", "..")
use(parts...) = Base.include(Main, joinpath(ROOT, parts...))
use("models", "ode", "ode.jl")

# --- Time grid and harmonic drive γ cos(ω t) entering the velocity equation ---
δt = 0.01; t_end = 300.0; ts = 0.0:δt:t_end
γ = 0.37; ω = 1.2
drive(t) = [0.0, γ * cos(ω * t)]

# --- Simulate the Duffing model over the specified time horizon ---
model = duffing_model()
X = Matrix{Float64}(undef, 2, length(ts)); X[:, 1] = [1.0, 0.0]
for i in 2:length(ts)
    X[:, i] = model.predict(model.params, X[:, i-1], drive(ts[i]), δt)
    i % 2000 == 0 && println("  Time-step $i / $(length(ts))")
end
x1, x2 = X[1, :], X[2, :]



# ============================== VISUALIZATION ===============================

# --- Static figure: (x₁, x₂) phase portrait and the position / drive time series ---
portrait = plot(x1, x2; xlabel = "x₁", ylabel = "x₂",
                title = "Duffing oscillator: phase portrait (x₁–x₂)", legend = false, lw = 1,
                lc = :viridis, line_z = ts, colorbar = false)
series = plot(ts, [x1 γ .* cos.(ω .* ts)]; xlabel = "t", title = "position and drive over time",
              label = ["x₁(t)" "γ cos(ω t)"], lw = 1)
plt = plot(portrait, series, layout = (1, 2), size = (1200, 500))

# --- Animated trajectory tracing the state through the (x₁, x₂) phase plane ---
step = 40                                            # frames every `step` time steps
frame_ids = 2:step:length(ts)
anim = @animate for (frame, k) in enumerate(frame_ids)
    frame % 50 == 0 && println("  frame $frame / $(length(frame_ids))")
    plot(x1[1:k], x2[1:k]; xlabel = "x₁", ylabel = "x₂",
         title = "Duffing oscillator, t = $(round(ts[k], digits = 1))",
         legend = false, lw = 1, lc = :viridis, line_z = ts[1:k], colorbar = false,
         xlims = extrema(x1), ylims = extrema(x2), size = (700, 600))
    scatter!([x1[k]], [x2[k]]; mc = :red, ms = 4)
end

# --- Store the figure and the animation in results/ode ---
resultdir = joinpath(ROOT, "results", "ode"); mkpath(resultdir)
savefig(plt, joinpath(resultdir, "Duffing_trajectory.png"))
gif(anim, joinpath(resultdir, "Duffing_trajectory.gif"), fps = 25)
