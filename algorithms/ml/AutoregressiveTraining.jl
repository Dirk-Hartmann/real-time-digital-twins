# Autoregressive (unrolled) training of a neural vector field — the neural-ODE objective.
# Instead of matching a precomputed derivative, the model is marched with explicit Euler
# across short chunks of trajectory and made to reproduce them. Each chunk is a segment
# of L consecutive states stored in `Xseg` (nin × B × L, B segments); starting from the
# first state of every segment the loss unrolls
#             x̃ⁿ⁺¹ = x̃ⁿ + δt f_θ(x̃ⁿ, uⁿ),   x̃⁰ = Xseg[:, :, 1],
# and sums the per-component-weighted mismatch against the true states along the chunk,
# with an L2 penalty on the weights,
#             L(θ) = mean over k of ‖ (x̃ᵏ - Xseg[:, :, k]) ./ σ ‖²  +  λ ‖θ‖²,
# backpropagating through every Euler step. Matching a trajectory rather than a pointwise
# derivative makes the predictor account for its own accumulating step error. An optional
# `Useg` (nctrl × B × L-1, columns aligned with the step transitions) supplies a per-step
# control sample for driven systems — each segment's batch column keeps its own control
# history, since different segments generally start at different times — and defaults to
# zero (autonomous systems need not pass it). The optimiser is an argument (`Adam` or
# `LBFGS`). `rollout_loss` is exposed separately so a script can track it on a validation
# set during training (via `Xseg_val` — recorded every iteration without entering the
# gradient) and evaluate it once more on a further held-out set afterwards.
#
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Optimization, OptimizationOptimisers, OptimizationOptimJL, Zygote, Statistics

# --- Rollout loss over trajectory chunks Xseg (nin × B × L), unrolling from the first state,
#     under an optional per-step control Useg (nctrl × B × L-1, zero if not supplied) ---
function rollout_loss(model, θ, Xseg, δt; Useg = nothing, σ = ones(size(Xseg, 1)), λ = 1e-4)
    u0 = zeros(size(Xseg, 1)); L = size(Xseg, 3)
    x = Xseg[:, :, 1]
    e = 0.0
    for k in 2:L
        u = Useg === nothing ? u0 : Useg[:, :, k-1]
        x = model.predict(θ, x, u, δt)
        e += mean(abs2, (x .- Xseg[:, :, k]) ./ σ)
    end
    return e / (L - 1) + λ * sum(abs2, θ)
end

# --- Fit f_θ by unrolling over trajectory chunks Xseg (optionally driven by Useg); returns
#     weights and loss history ---
function train_model_autoregressive(model, Xseg, δt; Useg = nothing, σ = ones(size(Xseg, 1)), λ = 1e-4, θ0 = model.params,
                                    optimizer = OptimizationOptimisers.Adam(1e-3), iters = 1500, log_every = 200,
                                    Xseg_val = nothing, Useg_val = nothing)
    history = Float64[]
    history_val = Float64[]
    cb = (state, l) -> (push!(history, l);
        Xseg_val !== nothing && push!(history_val, rollout_loss(model, state.u, Xseg_val, δt; Useg = Useg_val, σ = σ, λ = λ));
        log_every > 0 && length(history) % log_every == 0 &&
            println("  iter $(length(history)) / $iters   loss $(round(l, sigdigits = 4))");
        false)
    optf = OptimizationFunction((θ, _) -> rollout_loss(model, θ, Xseg, δt; Useg = Useg, σ = σ, λ = λ), Optimization.AutoZygote())
    res  = solve(OptimizationProblem(optf, θ0), optimizer; maxiters = iters, callback = cb)
    return Xseg_val === nothing ? (res.u, history) : (res.u, history, history_val)
end
