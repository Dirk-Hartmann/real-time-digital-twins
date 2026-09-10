# Operator inference for the PCB heat problem: learn a reduced dynamical model from
# data alone. One reference trajectory from the PCB training ensemble
# (data/PCB_training.bson) is first projected onto a POD basis; on the reduced
# coordinates a polynomial vector field
#             dz/dt ≈ A z + C u
# is fitted by matching first-order forward finite-difference derivative estimates
#             dz/dt(z_n) ≈ (z_{n+1} - z_n) / δt,
# posed as a one-shot least-squares problem in the operators θ = (A, C) - the classic
# operator-inference fit, mirroring scripts/ml/Lorenz_poly_finitediff.jl but with the
# POD reduction in front and the recorded drives u = [c; h] as control inputs. Since
# the full PCB system is linear, the linear model `polynomial_model_AC` (no quadratic
# H) suffices. The reduced heat modes are stiff, so an explicit-Euler rollout of the
# raw least-squares operator is unstable; a small Tikhonov L2 penalty (λ = 1e-4) tames
# the few ill-conditioned fast-mode directions and keeps the rollout stable without
# over-damping the energetic slow modes. The learned reduced model is then marched under
# the recorded drives and compared against the projected reference and the full field.
# The reduced coordinates are standardised beforehand so the modes, whose magnitudes
# decay with the POD spectrum, enter the regression on a common scale.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using LinearAlgebra, Statistics, Random, Plots, BSON

println("\nLearning a reduced operator-inference model for the PCB heat problem from data...")

# --- Axis tick labels rounded to 3 significant digits on all plots ---
default(formatter = y -> y isa Number ? string(round(y, sigdigits = 3)) : string(y))

# --- Include repo files relative to the root (avoids repeated joinpath calls) ---
const ROOT = normpath(@__DIR__, "..", "..")
use(parts...) = Base.include(Main, joinpath(ROOT, parts...))
use("util", "util.jl")
use("algorithms", "ml", "ml.jl")
use("algorithms", "rom", "rom.jl")

# --- Load one reference trajectory from the PCB training ensemble, drives, and grid ---
data = BSON.load(joinpath(ROOT, "data", "PCB_training.bson"))
sim = 1                                                # pick one simulation from the ensemble as reference
ts = data[:t]
c, h = data[:u][1, :, sim], data[:u][2, :, sim]
g = Grid(data[:nelx], data[:nely]; δx = data[:δx])
Xref = data[:T][:, :, sim]                             # n × m snapshot matrix
m = length(ts); δt = diff(ts)

# --- POD reduction and standardised reduced coordinates ζ = z ./ s ---
r = 8
red = pod_reduction(Xref; r = r)
Zref = red.encode(red.params, Xref)
s = vec(std(Zref, dims = 2)) .+ eps()
ζ = Zref ./ s

# --- Finite-difference derivative labels on the reduced coordinates, with drives ---
X = ζ[:, 1:m-1]
Y = (ζ[:, 2:m] .- ζ[:, 1:m-1]) ./ reshape(δt, 1, :)
U = permutedims(hcat(c[1:m-1], h[1:m-1]))              # 2 × (m-1) control samples

# --- Fit the reduced operators by one-shot least squares (operator inference) ---
model = polynomial_model_AC(; nin = r, nctrl = 2, rng = MersenneTwister(0))
θ = train_model_leastsquares(model, X, Y; U = U, λ = 1e-4)

# --- March the learned reduced model under the recorded drives (explicit Euler) ---
ζpred = Matrix{Float64}(undef, r, m); ζpred[:, 1] = ζ[:, 1]
for n in 2:m
    ζpred[:, n] = model.predict(θ, ζpred[:, n-1], [c[n-1], h[n-1]], δt[n-1])
end
Zpred = s .* ζpred
Xpred = red.decode(red.params, Zpred)                  # learned reduced prediction, full space
Xproj = red.decode(red.params, Zref)                   # POD reconstruction baseline

# --- Full-field relative error over time (prediction vs POD projection floor) ---
relerr(A) = [norm(Xref[:, n] - A[:, n]) / (norm(Xref[:, n]) + eps()) for n in 1:m]
err_pred, err_proj = relerr(Xpred), relerr(Xproj)
println("Operator inference  r = $r   mean pred err = $(round(mean(err_pred), sigdigits = 3))   POD projection err = $(round(mean(err_proj), sigdigits = 3))")



# ============================== VISUALIZATION ===============================

# --- Reshape final-time fields to (ny × nx) grids ---
Tfull = tomat(g, Xref[:, end]); Tpred = tomat(g, Xpred[:, end])
clims = extrema(Tfull)
hm(z, title; kw...) = heatmap(z; title, yflip = true, aspect_ratio = 1, axis = false, ticks = false, kw...)

# --- Visualise the ROM prediction field, its difference to the full solution, the ROM error
#     over time (projection vs prediction), and the reduced trajectories ---
p1 = plot(ts, Zref[1:3, :]'; lw = 2, label = ["z₁ true" "z₂ true" "z₃ true"],
          xlabel = "t [s]", ylabel = "reduced coord.", title = "POD coordinates: true vs learned")
plot!(p1, ts, Zpred[1:3, :]'; lw = 1.5, ls = :dash, label = ["z₁ learned" "z₂ learned" "z₃ learned"])
plt = plot(
    hm(Tpred,         "T operator inference (t = $(round(ts[end], digits = 1)) s, r = $r)"; color = :thermal, clims = clims),
    hm(Tfull - Tpred, "difference to full solution";                                        color = :balance),
    plot(ts[2:end], [err_proj[2:end] err_pred[2:end]]; yscale = :log10, xlabel = "t [s]", ylabel = "rel. error",
         title = "ROM error over time", label = ["POD projection" "learned prediction"], lw = 2),
    p1,
    layout = (2, 2), size = (1100, 900),
)

# --- Store the operator-inference figure in results/rom ---
resultdir = joinpath(ROOT, "results", "rom"); mkpath(resultdir)
savefig(plt, joinpath(resultdir, "PCB_OperatorInference.png"))
