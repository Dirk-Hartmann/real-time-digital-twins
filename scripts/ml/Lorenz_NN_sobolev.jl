# Learn a neural-network vector field f_θ(x) ≈ dx/dt for the Lorenz system from the
# training trajectories in data/Lorenz_training.bson with a SOBOLEV objective: besides
# the first-order forward finite-difference derivative label used in
# Lorenz_NN_finitediff.jl,
#             dx/dt(x_n) ≈ (x_{n+1} - x_n) / δt,
# the fit is also supervised with the TRUE Jacobian of the generating system,
# J_n = ∂f/∂x(x_n), evaluated analytically from the Lorenz model in
# models/ode/LorenzSystem.jl. The network is trained to match both the derivative and
# its own sensitivity to the state, `train_model_sobolev` penalising the mismatch of
# each. This is extra structural information a plain residual fit cannot use — since it
# is available here (the data-generating vector field is known), a Sobolev fit can
# exploit it for a tighter match from the same samples. It is the third of three
# comparable scripts (see also Lorenz_NN_finitediff.jl and Lorenz_NN_neuralode.jl), which
# share structure and differ only in their target / loss section.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Plots, BSON, Random, Statistics

println("\nFitting a neural vector field to the Lorenz system with a Sobolev objective (derivative + true Jacobian)...\n")

# --- Include repo files relative to the root (avoids repeated joinpath calls) ---
const ROOT = normpath(@__DIR__, "..", "..")
use(parts...) = Base.include(Main, joinpath(ROOT, parts...))
use("util", "util.jl")
use("models", "ode", "ode.jl")
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

# --- Jacobian labels: the TRUE Lorenz Jacobian ∂f/∂x at every training sample ---
u0 = zeros(3)                                        # zero forcing, used both for labels and marching
lorenz = lorenz_model()
jacobian_labels(trajs) = stack(lorenz.jacobian(lorenz.params, X[:, i, k], u0) for k in trajs for i in idx)
J, J_val = jacobian_labels(train), jacobian_labels(validation)

# --- Build the neural model (fixed seed and architecture shared across the scripts) ---
model = neural_model(; width = 32, depth = 2, μ = μ, σ = σ, rng = MersenneTwister(0))
opt = OptimizationOptimisers.Adam(2e-2)              # or OptimizationOptimJL.LBFGS() for a quasi-Newton method
λ = 1e-4                                             # L2 weight penalty, shared by the training and final losses
α_jac = 1.0                                          # weight of the Jacobian term in the Sobolev loss

# --- Train: regress f_θ onto Y and its Jacobian onto J, tracking the validation loss alongside ---
θ, history, history_val = train_model_sobolev(model, Xb, Y, J; Xval = Xb_val, Yval = Y_val, Jval = J_val,
                                                σ = σ, α = α_jac, λ = λ, optimizer = opt, iters = 10000)
println("Final losses — train: $(round(history[end], sigdigits = 4))   ",
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
           plot_title = "Lorenz System:  Sobolev NN regression (derivative + Jacobian)")

# --- Store the figure ---
resultdir = joinpath(ROOT, "results", "ml"); mkpath(resultdir)
savefig(plt, joinpath(resultdir, "Lorenz_NN_sobolev.png"))
