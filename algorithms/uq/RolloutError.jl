# Rollout-error scoring for any model exposing the `predict(θ, x, u, δt)` interface (ODE,
# reduced-order, or learned models alike). Trajectories are split into non-overlapping
# chunks of L consecutive raw time steps; from each chunk's initial condition x0 the model
# is rolled out for L explicit-Euler steps and scored against the true continuation by the
# mean squared error,
#             mse(x0) = mean ‖ x̃^{1:L} - x^{s+1:s+L} ‖²,   x0 = x^s,
# where x̃ is the model's own L-step rollout from x0. Pairing every chunk's initial condition
# with its rollout error, (x0_n, mse_n), gives scattered samples of the model's OWN prediction
# accuracy as a function of where it started — the raw material for the Gaussian-process
# accuracy map fit in GPErrorModel.jl. Autonomous models pass a constant `u` (the default,
# zero); DRIVEN models (e.g. a PCB reduced-order model under recorded drives) instead pass
# the full per-step, per-trajectory control trajectory `U` (nctrl × T × ntraj, aligned with
# X), from which the L-step control window of every chunk is sliced automatically.
#
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Statistics

# --- Roll the model out for L explicit-Euler steps from x0 under control U (a constant vector,
#     or a matrix with one column per step); returns the (nin × L+1) trajectory ---
function rollout_steps(model, θ, x0, U, δt, L)
    X = Array{Float64}(undef, length(x0), L + 1)
    X[:, 1] = x0
    for n in 2:L+1
        u = U isa AbstractVector ? U : U[:, n-1]
        X[:, n] = model.predict(θ, X[:, n-1], u, δt)
    end
    return X
end

# --- Mean squared L-step rollout error at every chunk of trajectories `trajs` (indices into
#     X's third dimension), starting every `stride` steps (default L, i.e. non-overlapping —
#     use a smaller stride for a denser accuracy map when few trajectories are available);
#     returns the chunk initial conditions X0 (nin × N) and their errors mse (N). Autonomous
#     models use the constant `u` (default zero); driven models pass the full control
#     trajectory `U` (nctrl × T × ntraj, aligned with X) instead ---
function rollout_chunk_errors(model, θ, X, δt, trajs; L = 5, stride = L, u = zeros(size(X, 1)), U = nothing)
    T = size(X, 2)
    s0 = 1:stride:(T - L)
    starts = [(k, s) for k in trajs for s in s0]
    X0  = reduce(hcat, [X[:, s, k] for (k, s) in starts])
    control(k, s) = U === nothing ? u : U[:, s:s+L-1, k]
    mse = [mean(abs2, rollout_steps(model, θ, X[:, s, k], control(k, s), δt, L)[:, 2:end] .- X[:, s:s+L, k][:, 2:end])
           for (k, s) in starts]
    return X0, mse
end
