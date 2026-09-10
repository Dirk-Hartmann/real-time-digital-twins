# Uncertainty quantification for the neural vector field trained in Lorenz_NN_finitediff.jl:
# a Gaussian-process surrogate learns to predict, FROM THE CURRENT STATE ALONE, how large the
# model's own 5-step rollout error will be,  an "accuracy map" that turns a single trained
# model into one with a calibrated, state-dependent uncertainty estimate.
#
# The neural model is trained exactly as in Lorenz_NN_finitediff.jl. Its training trajectories
# are then split into non-overlapping chunks of L = 5 raw time steps; for every chunk the model
# is rolled out over those 5 steps from the chunk's initial condition and scored against the
# true continuation by the mean squared error (`rollout_chunk_errors`,
# algorithms/uq/RolloutError.jl). This gives scattered samples (x0_n, mse_n) - the model's own
# rollout error, indexed by where it started - which are fit by a Gaussian process
# (`fit_error_gp`, algorithms/uq/GPErrorModel.jl) mapping state -> (mean, std) of the expected
# error. On a held-out validation trajectory, the true 5-step rollout error at every chunk is
# then compared against the GP's mean/uncertainty band, predicted from the state alone.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Plots, BSON, Random, Statistics

println("\nFitting a Gaussian-process error estimator that predicts the Lorenz neural model's prediction error...")

# --- Include repo files relative to the root (avoids repeated joinpath calls) ---
const ROOT = normpath(@__DIR__, "..", "..")
use(parts...) = Base.include(Main, joinpath(ROOT, parts...))
use("util", "util.jl")
use("algorithms", "ml", "ml.jl")
use("algorithms", "uq", "uq.jl")

# --- Load the training trajectories  (X is 3 × T × ntraj, sampled at step δt) ---
BSON.@load joinpath(ROOT, "data", "Lorenz_training.bson") t X
δt = t[2] - t[1]; T = size(X, 2)
val_frac = 0.10; test_frac = 0.0                     # no held-out test set; training gets the remainder
stride = 4
train, validation, _ = split_trajectories(size(X, 3); val_frac = val_frac, test_frac = test_frac)

# --- Standardise the state from the training data (fed to the network as constants) ---
μ = vec(mean(reshape(X[:, :, train], 3, :); dims = 2))
σ = vec(std(reshape(X[:, :, train], 3, :); dims = 2))

# --- Derivative labels: first-order forward finite difference (train / validation splits) ---
idx = 1:stride:(T - 1)
finite_diff_labels(trajs) = reshape(X[:, idx, trajs], 3, :), reshape((X[:, idx .+ 1, trajs] .- X[:, idx, trajs]) ./ δt, 3, :)
Xb, Y         = finite_diff_labels(train)
Xb_val, Y_val = finite_diff_labels(validation)

# --- Train the neural model, exactly as in Lorenz_NN_finitediff.jl ---
model = neural_model(; width = 32, depth = 2, μ = μ, σ = σ, rng = MersenneTwister(0))
u0 = zeros(3)
opt = OptimizationOptimisers.Adam(2e-2)
λ = 1e-4
θ, history, history_val = train_model_residual(model, Xb, Y; Xval = Xb_val, Yval = Y_val,
                                                  σ = σ, λ = λ, optimizer = opt, iters = 10000)
println("Neural model trained — final train loss $(round(history[end], sigdigits = 4)), | ",
        "validation loss $(round(history_val[end], sigdigits = 4))")

# --- Accuracy map: 5-step rollout error at every chunk of the TRAINING trajectories ---
L = 5
X0, mse = rollout_chunk_errors(model, θ, X, δt, train; L = L, u = u0)
println("Accuracy map: $(length(mse)) chunks, mean rollout MSE $(round(mean(mse), sigdigits = 4))")

# --- Fit the Gaussian-process error surrogate x -> (mean, std) of the 5-step rollout MSE ---
gpmodel = fit_error_gp(X0, mse; rng = MersenneTwister(1))

# --- Demo on a held-out validation trajectory: true vs GP-estimated rollout error at every chunk ---
demo_traj = validation[1]
X0_demo, true_err = rollout_chunk_errors(model, θ, X, δt, [demo_traj]; L = L, u = u0)
gp_mean, gp_std   = predict_error(gpmodel, X0_demo)
te = t[1:L:(T - L)]                                  # chunk start times, aligned with X0_demo's order

# --- Fully simulate the model over that same trajectory's whole horizon, for comparison ---
Xpred_demo = rollout_steps(model, θ, X[:, 1, demo_traj], u0, δt, T - 1)




# ============================== VISUALIZATION ===============================

# --- The validation trajectory: true vs fully-simulated model, one component per panel ---
p3 = plot(t, X[1, :, demo_traj]; xlabel = "t", ylabel = "x", title = "validation trajectory — x(t)", lc = :black, label = "true")
plot!(p3, t, Xpred_demo[1, :]; lc = :crimson, ls = :dash, label = "predicted")
p4 = plot(t, X[2, :, demo_traj]; xlabel = "t", ylabel = "y", title = "validation trajectory — y(t)", lc = :black, label = "true")
plot!(p4, t, Xpred_demo[2, :]; lc = :crimson, ls = :dash, label = "predicted")
p5 = plot(t, X[3, :, demo_traj]; xlabel = "t", ylabel = "z", title = "validation trajectory — z(t)", lc = :black, label = "true")
plot!(p5, t, Xpred_demo[3, :]; lc = :crimson, ls = :dash, label = "predicted")

# --- Figure: the accuracy map (training chunks) and the true-vs-GP error comparison ---
p1 = scatter(X0[1, :], X0[3, :]; zcolor = log10.(mse .+ 1e-10), ms = 2, msw = 0,
             colorbar_title = "log10(rollout MSE)", xlabel = "x", ylabel = "z",
             title = "accuracy map (training chunks)", label = false)
p2 = plot(te, gp_mean; ribbon = 2 .* gp_std, lw = 1.5, lc = :crimson, label = "GP estimate (±2σ)",
          xlabel = "t", ylabel = "5-step rollout MSE", title = "validation trajectory")
plot!(p2, te, true_err; lw = 1.5, lc = :black, label = "true error")

plt = plot(p3, p4, p5, p1, p2; layout = (2, 3), size = (1500, 900),
           plot_title = "Lorenz attractor: Gaussian-process error estimator")

# --- Store the figure ---
resultdir = joinpath(ROOT, "results", "uq"); mkpath(resultdir)
savefig(plt, joinpath(resultdir, "Lorenz_NN_GP-ErrorEstimator.png"))
