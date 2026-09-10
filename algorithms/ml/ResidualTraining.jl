# Residual (derivative-matching) training of a neural vector field. Given state samples
# x_n and matching estimates d_n ≈ dx/dt(x_n) of the time derivative — obtained, for
# instance, from a finite-difference stencil applied to sampled trajectories — the
# weights are fitted by minimising the per-component-weighted residual with an L2 penalty
#             L(θ) = mean ‖ (f_θ(x_n, u_n) - d_n) ./ σ ‖²  +  λ ‖θ‖²,
# a plain regression that never rolls the model forward. Control samples u_n enter
# through the optional `U` (its columns aligned with those of `X`) and default to zero,
# so autonomous systems need not pass it. The scaling σ balances the state components,
# whose derivative magnitudes differ, and λ ‖θ‖² regularises the weights. The optimiser
# is an argument, so the same loss can be minimised with a stochastic first-order method
# (`Adam`) or a quasi-Newton one (`LBFGS`). `residual_loss` is exposed separately so a
# script can track it on a validation set during training (via `Xval, Yval` — recorded
# every iteration without entering the gradient) and evaluate it once more on a further
# held-out set afterwards.
#
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Optimization, OptimizationOptimisers, OptimizationOptimJL, Zygote, Statistics

# --- Residual loss at states X against derivative estimates Y (with optional control U) ---
residual_loss(model, θ, X, Y; U = zeros(size(X, 1)), σ = ones(size(X, 1)), λ = 1e-4) =
    mean(abs2, (model.rhs(θ, X, U) .- Y) ./ σ) + λ * sum(abs2, θ)

# --- Fit f_θ to derivative estimates Y at states X (with optional control U); returns weights and loss history ---
function train_model_residual(model, X, Y; U = zeros(size(X, 1)), σ = ones(size(X, 1)), λ = 1e-4,
                              θ0 = model.params, optimizer = OptimizationOptimisers.Adam(1e-3), iters = 1500, log_every = 200,
                              Xval = nothing, Yval = nothing, Uval = zeros(size(X, 1)))
    history = Float64[]
    history_val = Float64[]
    cb = (state, l) -> (push!(history, l);
        Xval !== nothing && push!(history_val, residual_loss(model, state.u, Xval, Yval; U = Uval, σ = σ, λ = λ));
        log_every > 0 && length(history) % log_every == 0 &&
            println("  iter $(length(history)) / $iters   loss $(round(l, sigdigits = 4))");
        false)
    optf = OptimizationFunction((θ, _) -> residual_loss(model, θ, X, Y; U = U, σ = σ, λ = λ), Optimization.AutoZygote())
    res  = solve(OptimizationProblem(optf, θ0), optimizer; maxiters = iters, callback = cb)
    return Xval === nothing ? (res.u, history) : (res.u, history, history_val)
end
