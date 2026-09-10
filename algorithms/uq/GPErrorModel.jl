# Gaussian-process surrogate for the LOCAL prediction error of a trained vector-field model.
# Given scattered pairs (x0_n, e_n) — an initial condition and the mean-squared rollout error
# a trained model incurs starting from it (e.g. from `rollout_chunk_errors` in
# RolloutError.jl) — a Gaussian process regresses e(x) ≈ GP(x), providing both a MEAN estimate
# of the expected error at any new state x and a VARIANCE quantifying the uncertainty of that
# estimate. This is the "accuracy map" of the trained model: cheap to query pointwise, it turns
# a discrete set of rollout experiments into a continuous error/uncertainty field over state
# space, without touching the underlying model's own weights.
#
# Built on GaussianProcesses.jl: a squared-exponential ARD kernel (one length scale per state
# component, initialised from the spread of the training states) with a constant mean and
# hyperparameters fit by maximum likelihood. An EXACT GP's covariance matrix is dense (N × N)
# and its fit costs O(N^3), so `fit_error_gp` randomly subsamples down to `max_points` chunks
# before fitting whenever there are more (rollout chunks routinely number in the tens of
# thousands, far beyond what a dense Cholesky factorisation can handle). The GP is fit to
# log(e) rather than e itself, since a mean-squared error is strictly positive and typically
# heavily right-skewed (many near-perfect chunks, a long tail of hard ones) — fitting the raw
# values makes the noise/length-scale optimisation numerically unstable (near-singular
# covariance matrices). `predict_error` converts the fitted log-space (mean, variance) back to
# the error's own scale via the standard log-normal moment formulas.
#
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using GaussianProcesses, Statistics, LinearAlgebra, PDMats, Random

# GaussianProcesses v0.12's own `ldiv!(::PDMat, x)` is ambiguous with a newer PDMats.jl's
# generic method; both resolve the same way, so disambiguate by delegating to the Cholesky factor.
LinearAlgebra.ldiv!(cK::PDMat, x::AbstractVecOrMat) = ldiv!(cK.chol, x)

struct ErrorGPModel
    gp     # GaussianProcesses.GP object fit to (x0, mse) pairs
end

# --- Fit a GP surrogate (in log space) x -> expected rollout error, from initial conditions
#     X0 (nin × N) and their errors mse (N), subsampled to at most `max_points`; hyperparameters
#     are optimised by maximum likelihood unless opt = false ---
function fit_error_gp(X0, mse; opt = true, max_points = 1000, rng = Random.default_rng())
    if length(mse) > max_points
        keep = randperm(rng, length(mse))[1:max_points]
        X0, mse = X0[:, keep], mse[keep]
    end
    y  = log.(mse .+ 1e-10)                             # strictly positive, less heavy-tailed target
    ll = log.(vec(std(X0; dims = 2)) .+ eps())          # initial length scale ~ state spread, per component
    kernel = SEArd(ll, 0.0)
    gp = GP(X0, y, MeanConst(mean(y)), kernel, log(std(y) / 10 + eps()))
    opt && optimize!(gp)
    return ErrorGPModel(gp)
end

# --- Predict the expected error and its standard deviation at states X (nin × N), converting
#     the GP's log-space (mean, variance) back to the error's own scale (log-normal moments) ---
function predict_error(m::ErrorGPModel, X)
    μ, σ2 = predict_y(m.gp, X)
    mean_e = exp.(μ .+ σ2 ./ 2)
    var_e  = (exp.(σ2) .- 1) .* exp.(2 .* μ .+ σ2)
    return mean_e, sqrt.(var_e)
end
