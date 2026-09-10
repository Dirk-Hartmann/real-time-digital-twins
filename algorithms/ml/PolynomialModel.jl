# Polynomial vector fields for data-driven prediction of a controlled ODE. Two models
# are provided, differing only in whether a quadratic term is present:
#
#   polynomial_model_AC   :  f_θ(x, u) = A x        + C u     (linear + control)
#   polynomial_model_AHC  :  f_θ(x, u) = A x + H q(x) + C u   (linear + quadratic + control)
#
# The linear term A x and control term C u are shared; the AHC model adds the quadratic
# term H q(x). The quadratic feature vector q(x) = [ x_i x_j : i ≤ j ] collects the
# UNIQUE degree-two monomials of the state (n(n+1)/2 of them), so every unordered pair
# {i, j} carries a single coefficient: the quadratic operator is kept in its symmetric
# form and the two orderings share it, H_ij x_i x_j == H_ji x_j x_i, with no double
# counting.
#
# Both share the `PolynomialModel` struct, and both are LINEAR in their coefficients θ,
# so besides `rhs` and `predict` they expose a `features` map returning the stacked
# regressor φ(x, u), [x; u] for AC, [x; q(x); u] for AHC, whose blocks line up with
# the coefficient blocks. The same fit can therefore be posed either as a gradient-based
# regression (ResidualTraining.jl, AutoregressiveTraining.jl) or as a one-shot least-
# squares solve (LeastSquaresTraining.jl). Marching f_θ with one explicit Euler step
# gives the predictor predict(θ, x, u, δt) = x + δt f_θ(x, u), so a `PolynomialModel`
# shares the `ODEModel` / `NeuralModel` / `ReducedModel` interface.
#
# All maps act column-wise, so x may be a single state or a batch of states, and the
# control u is broadcast across a batch. For a bilinear system such as Lorenz the
# quadratic AHC model reproduces the dynamics exactly; for a linear system the AC model
# alone recovers the reduced operators by operator inference.
#
# ----------------------------------------------------------------------------------
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Random, ComponentArrays

struct PolynomialModel
    params    # ComponentArray of coefficient matrices  (A, C) or (A, H, C)
    rhs       # (θ, x, u)      -> f_θ(x, u)
    predict   # (θ, x, u, δt)  -> x + δt f_θ(x, u)
    features  # (x, u)         -> stacked regressor φ(x, u) for the least-squares fit
end

# --- Unique quadratic monomials q(x) = [x_i x_j : i ≤ j] (batched over columns) ---
quad_features(x::AbstractVector, pairs) = [x[i] * x[j] for (i, j) in pairs]
quad_features(x::AbstractMatrix, pairs) = reduce(vcat, [reshape(x[i, :] .* x[j, :], 1, :) for (i, j) in pairs])

# --- Broadcast the control across a batch so it stacks with the (batched) state ---
control(x::AbstractVector, u) = u
control(x::AbstractMatrix, u::AbstractVector) = repeat(u, 1, size(x, 2))
control(x::AbstractMatrix, u::AbstractMatrix) = u

# --- Linear model:  f_θ(x, u) = A x + C u ---
function polynomial_model_AC(; nin = 3, nctrl = nin, init = 1e-2, rng = Random.default_rng())
    params = ComponentArray(A = init .* randn(rng, nin, nin),
                            C = init .* randn(rng, nin, nctrl))
    rhs(θ, x, u)      = θ.A * x .+ θ.C * control(x, u)
    predict(θ, x, u, δt) = x .+ δt .* rhs(θ, x, u)
    features(x, u)    = vcat(x, control(x, u))                # φ = [x; u]
    return PolynomialModel(params, rhs, predict, features)
end

# --- Quadratic model:  f_θ(x, u) = A x + H q(x) + C u  (symmetric H) ---
function polynomial_model_AHC(; nin = 3, nctrl = nin, init = 1e-2, rng = Random.default_rng())
    pairs = [(i, j) for i in 1:nin for j in i:nin]            # unique pairs -> symmetric H
    params = ComponentArray(A = init .* randn(rng, nin, nin),
                            H = init .* randn(rng, nin, length(pairs)),
                            C = init .* randn(rng, nin, nctrl))
    rhs(θ, x, u)      = θ.A * x .+ θ.H * quad_features(x, pairs) .+ θ.C * control(x, u)
    predict(θ, x, u, δt) = x .+ δt .* rhs(θ, x, u)
    features(x, u)    = vcat(x, quad_features(x, pairs), control(x, u))   # φ = [x; q(x); u]
    return PolynomialModel(params, rhs, predict, features)
end
