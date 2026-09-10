# Sobolev (derivative-AND-Jacobian-matching) training of a neural vector field. Like
# ResidualTraining.jl, state samples x_n are matched against derivative estimates
# d_n ≈ dx/dt(x_n); Sobolev training additionally supervises the field's own sensitivity
# with target Jacobians J_n ≈ ∂f/∂x(x_n) — typically the analytic Jacobian of the system
# that generated the data, wherever one is known — penalising the mismatch of both,
#             L(θ) = mean ‖ (f_θ(x_n, u_n) - d_n) ./ σ ‖²
#                  + α  mean ‖ (J_θ(x_n, u_n) - J_n) ./ σ ‖²  +  λ ‖θ‖²,
# where J_θ(x, u) = ∂f_θ/∂x is itself estimated by central finite differences of the
# model's own `rhs`, mirroring how the label d_n is typically formed from data. Matching
# a Jacobian in addition to a function value — "Sobolev training" (Czarnecki et al.,
# 2017) — lets the fit exploit extra structural information whenever an accurate
# Jacobian target is available, improving on plain residual matching from the same
# samples; setting α = 0 recovers `residual_loss` exactly. Control samples u_n enter as
# in ResidualTraining.jl, through the optional `U`, and default to zero. The optimiser is
# an argument, and `sobolev_loss` is exposed separately so a script can track it on a
# validation set during training (via `Xval, Yval, Jval`) and evaluate it once more on a
# further held-out set afterwards.
#
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Optimization, OptimizationOptimisers, OptimizationOptimJL, Zygote, Statistics

# --- Central finite-difference Jacobian ∂f_θ/∂x at states X (nin × B), stacked as nin × nin × B ---
function model_jacobian(model, θ, X, U; ε = 1e-4)
    nin = size(X, 1)
    onehot(j) = [Float64(i == j) for i in 1:nin]
    cols = [(model.rhs(θ, X .+ ε .* onehot(j), U) .- model.rhs(θ, X .- ε .* onehot(j), U)) ./ 2ε for j in 1:nin]
    return permutedims(cat(cols...; dims = 3), (1, 3, 2))          # nin(out) × nin(in) × B
end

# --- Sobolev loss: residual against Y plus a weighted Jacobian mismatch against Jtarget ---
sobolev_loss(model, θ, X, Y, Jtarget; U = zeros(size(X, 1)), σ = ones(size(X, 1)), α = 1.0, λ = 1e-4, ε = 1e-4) =
    mean(abs2, (model.rhs(θ, X, U) .- Y) ./ σ) +
    α * mean(abs2, (model_jacobian(model, θ, X, U; ε = ε) .- Jtarget) ./ σ) +
    λ * sum(abs2, θ)

# --- Fit f_θ to derivative estimates Y and Jacobian targets Jtarget at states X (with optional
#     control U); returns weights and loss history ---
function train_model_sobolev(model, X, Y, Jtarget; U = zeros(size(X, 1)), σ = ones(size(X, 1)), α = 1.0, λ = 1e-4, ε = 1e-4,
                             θ0 = model.params, optimizer = OptimizationOptimisers.Adam(1e-3), iters = 1500, log_every = 200,
                             Xval = nothing, Yval = nothing, Jval = nothing, Uval = zeros(size(X, 1)))
    history = Float64[]
    history_val = Float64[]
    cb = (state, l) -> (push!(history, l);
        Xval !== nothing && push!(history_val, sobolev_loss(model, state.u, Xval, Yval, Jval; U = Uval, σ = σ, α = α, λ = λ, ε = ε));
        log_every > 0 && length(history) % log_every == 0 &&
            println("  iter $(length(history)) / $iters   loss $(round(l, sigdigits = 4))");
        false)
    optf = OptimizationFunction((θ, _) -> sobolev_loss(model, θ, X, Y, Jtarget; U = U, σ = σ, α = α, λ = λ, ε = ε), Optimization.AutoZygote())
    res  = solve(OptimizationProblem(optf, θ0), optimizer; maxiters = iters, callback = cb)
    return Xval === nothing ? (res.u, history) : (res.u, history, history_val)
end
