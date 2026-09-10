# Learn a quadratic polynomial vector field f_θ(x) = A x + H q(x) ≈ dx/dt for the Lorenz
# system from the training trajectories in data/Lorenz_training.bson with an AUTOREGRESSIVE
# (neural-ODE) objective: no precomputed derivative label is used. Instead the model is
# marched with explicit Euler over a short horizon,
#             x_{n+1} = x_n + δt f_θ(x_n),
# and the loss compares the whole rolled-out segment against the true states, back-
# propagating through every Euler step. Matching a trajectory rather than a pointwise
# derivative makes the predictor robust to its own accumulating step error. The parameters
# θ = (A, H, C) are exactly the entries of the polynomial coefficient matrices; the Lorenz
# field is linear plus bilinear, so this model class can reproduce it (near) exactly. The
# learned model is then marched further and compared against a held-out true trajectory.
# This mirrors Lorenz_NN_neuralode.jl and differs only in the model built.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Plots, BSON, Random, Statistics

println("\nFitting a polynomial vector field to the Lorenz system via an autoregressive (neural-ODE) rollout objective...\n")

# --- Include repo files relative to the root (avoids repeated joinpath calls) ---
const ROOT = normpath(@__DIR__, "..", "..")
use(parts...) = Base.include(Main, joinpath(ROOT, parts...))
use("util", "util.jl")
use("algorithms", "ml", "ml.jl")

# --- Load the training trajectories  (X is 3 × T × ntraj, sampled at step δt) ---
BSON.@load joinpath(ROOT, "data", "Lorenz_training.bson") t X
δt = t[2] - t[1]; T = size(X, 2)
val_frac = 0.10; test_frac = 0.0                     # no held-out test set; training gets the remainder
train, validation, _ = split_trajectories(size(X, 3); val_frac = val_frac, test_frac = test_frac)

# --- Component scaling of the loss (balances the differing state magnitudes) ---
σ = vec(std(reshape(X[:, :, train], 3, :); dims = 2))

# --- Training chunks: segments of L = K+1 consecutive states (nin × B × L), train / validation ---
K = 10; seg = 4                                      # chunk length and stride between starts
s0 = 1:seg:(T - K)
segments(trajs) = cat((reshape(X[:, s0 .+ k, trajs], 3, :) for k in 0:K)...; dims = 3)
Xseg     = segments(train)
Xseg_val = segments(validation)

# --- Build the polynomial model (linear + quadratic + control terms) ---
model = polynomial_model_AHC(; nin = 3, rng = MersenneTwister(0))
u0 = zeros(3)                                        # zero forcing, used when marching for evaluation
opt = OptimizationOptimisers.Adam(2e-2)             # or OptimizationOptimJL.LBFGS() for a quasi-Newton method
λ = 0e-6                                             # L2 weight penalty, shared by the training and final losses

# --- Train: unroll the model over the trajectory chunks and match them, tracking the validation loss alongside ---
θ, history, history_val = train_model_autoregressive(model, Xseg, δt; Xseg_val = Xseg_val,
                                                       σ = σ, λ = λ, optimizer = opt, iters = 5000)
println("Final losses — train: $(round(history[end], sigdigits = 4)) | ",
          "validation: $(round(history_val[end], sigdigits = 4))")

# --- Evaluate: march the learned model forward from every held-out initial condition ---
Nh = 300                                             # short horizon (chaos limits long-term match)
held_out = validation
Xtrue = X[:, 1:Nh + 1, held_out]
Xpred = Array{Float64}(undef, 3, Nh + 1, length(held_out)); Xpred[:, 1, :] = X[:, 1, held_out]
for n in 2:Nh + 1, k in eachindex(held_out)
    Xpred[:, n, k] = model.predict(θ, Xpred[:, n-1, k], u0, δt)
end
te = (0:Nh) .* δt

# --- Rollout loss, averaged over all held-out trajectories (not just the ones plotted below) ---
rollout_loss_val = mean(abs2, (Xpred .- Xtrue) ./ σ)
println("Rollout loss over $(length(held_out)) held-out trajectories: $(round(rollout_loss_val, sigdigits = 4))")




# ============================== VISUALIZATION ===============================

# --- Figure: phase projection, x component, and the training loss history ---
p1 = plot(; xlabel = "x", ylabel = "z", title = "x–z projection")
p2 = plot(; xlabel = "t", ylabel = "x", title = "x component")
for k in eachindex(validation)
    lbl = k == 1 ? "true" : false
    plot!(p1, Xtrue[1, :, k], Xtrue[3, :, k]; lw = 2, lc = :black, label = lbl)
    plot!(p2, te, Xtrue[1, :, k]; lw = 2, lc = :black, label = lbl)
end
for k in eachindex(validation)
    lbl = k == 1 ? "learned" : false
    plot!(p1, Xpred[1, :, k], Xpred[3, :, k]; lw = 1.5, lc = :crimson, ls = :dash, label = lbl)
    plot!(p2, te, Xpred[1, :, k]; lw = 1.5, lc = :crimson, ls = :dash, label = lbl)
end
p3 = plot(history; yscale = :log10, lw = 1.5, label = "train",
          xlabel = "iteration", ylabel = "loss", title = "training loss")
plot!(p3, history_val; yscale = :log10, lw = 1.5, label = "validation")
plt = plot(p1, p2, p3; layout = (1, 3), size = (1400, 450),
           plot_title = "Lorenz System: autoregressive polynomial regression")

# --- Store the figure ---
resultdir = joinpath(ROOT, "results", "ml"); mkpath(resultdir)
savefig(plt, joinpath(resultdir, "Lorenz_poly_autoregressive.png"))
