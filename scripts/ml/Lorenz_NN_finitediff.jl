# Learn a neural-network vector field f_θ(x) ≈ dx/dt for the Lorenz system from the
# training trajectories in data/Lorenz_training.bson, using a FIRST-ORDER forward
# finite difference as the derivative label:
#             dx/dt(x_n) ≈ (x_{n+1} - x_n) / δt.
# The network regresses f_θ(x_n) onto that estimate; the learned model is then marched
# with explicit Euler and compared against a held-out true trajectory. This is the
# baseline of the two comparable scripts (see also Lorenz_NN_neuralode.jl), which share
# structure and differ only in their target / loss section.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Plots, BSON, Random, Statistics

println("\nFitting a neural vector field to the Lorenz system via finite-difference derivatives...\n")

# --- Include repo files relative to the root (avoids repeated joinpath calls) ---
const ROOT = normpath(@__DIR__, "..", "..")
use(parts...) = Base.include(Main, joinpath(ROOT, parts...))
use("util", "util.jl")
use("algorithms", "ml", "ml.jl")

# --- Load the training trajectories  (X is 3 × T × ntraj, sampled at step δt) ---
BSON.@load joinpath(ROOT, "data", "Lorenz_training.bson") t X
δt = t[2] - t[1]; T = size(X, 2)
val_frac = 0.10; test_frac = 0.0                     # no held-out test set; training gets the remainder
stride = 4                                           # subsample time to keep the batch small
train, validation, _ = split_trajectories(size(X, 3); val_frac = val_frac, test_frac = test_frac)

# --- Standardise the state from the training data (fed to the network as constants) ---
μ = vec(mean(reshape(X[:, :, train], 3, :); dims = 2))
σ = vec(std(reshape(X[:, :, train], 3, :); dims = 2))

# --- Derivative labels: first-order forward finite difference (train / validation splits) ---
idx = 1:stride:(T - 1)
finite_diff_labels(trajs) = reshape(X[:, idx, trajs], 3, :), reshape((X[:, idx .+ 1, trajs] .- X[:, idx, trajs]) ./ δt, 3, :)
Xb, Y         = finite_diff_labels(train)
Xb_val, Y_val = finite_diff_labels(validation)

# --- Build the neural model (fixed seed and architecture shared across the scripts) ---
model = neural_model(; width = 32, depth = 2, μ = μ, σ = σ, rng = MersenneTwister(0))
u0 = zeros(3)                                        # zero forcing, used when marching for evaluation
opt = OptimizationOptimisers.Adam(2e-2)              # or OptimizationOptimJL.LBFGS() for a quasi-Newton method
λ = 1e-4                                             # L2 weight penalty, shared by the training and final losses

# --- Train: regress f_θ onto the derivative estimates Y, tracking the validation loss alongside ---
θ, history, history_val = train_model_residual(model, Xb, Y; Xval = Xb_val, Yval = Y_val,
                                                  σ = σ, λ = λ, optimizer = opt, iters = 10000)
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
           plot_title = "Lorenz System: finite difference NN regression")

# --- Store the figure ---
resultdir = joinpath(ROOT, "results", "ml"); mkpath(resultdir)
savefig(plt, joinpath(resultdir, "Lorenz_NN_finitediff.png"))
