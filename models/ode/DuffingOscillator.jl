# Duffing oscillator, a forced nonlinear (double-well) oscillator
#             d²x/dt² + δ dx/dt + α x + β x³ = drive(t)
# written for the state vector x = [x₁, x₂] (position, velocity) as
#             dx₁/dt = x₂
#             dx₂/dt = -α x₁ - β x₁³ - δ x₂ + u₂
# i.e. dx/dt = f(x) + u, where the external forcing u = [0, drive(t)] carries the
# harmonic drive γ cos(ω t) supplied per step by the caller (the model itself is the
# unforced oscillator). For a double well (α < 0, β > 0) with light damping and a
# resonant drive the response is chaotic, tracing a strange attractor in the (x₁, x₂)
# phase plane.
#
# The model uses the `ODEModel` interface, i.e., it bundles its `params` with maps that take
# those parameters explicitly as their first argument. Time stepping uses a linearly-implicit
# (Rosenbrock-type) trapezoidal step built from the true Jacobian A(x) = ∂f/∂x (the cubic
# stiffness β x₁³ differentiates to 3β x₁²),
#             (I - δt/2 A(x*)) ⋅ (xⁿ⁺¹ - xⁿ) = δt (f(xⁿ) + u),
# in which the next state enters only linearly, so each step is a single 2×2 solve
# with no Newton iteration. Freezing A at the explicit midpoint predictor
# x* = xⁿ + δt/2 (f(xⁿ) + u) keeps the scheme second-order accurate.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt


using LinearAlgebra

# --- Definition of the Duffing oscillator ---
duffing_rhs(p, x, u) = [ x[2],
                        -p.α * x[1] - p.β * x[1]^3 - p.δ * x[2] ] .+ u

duffing_jacobian(p, x) = [ 0.0                          1.0
                          -(p.α + 3 * p.β * x[1]^2)    -p.δ ]

# --- One linearly-implicit trapezoidal step (A frozen at the midpoint predictor) ---
function duffing_predict(p, x, u, δt)
    f = duffing_rhs(p, x, u)
    x_mid = x .+ (δt / 2) .* f
    A = duffing_jacobian(p, x_mid)
    return x .+ (I - (δt / 2) .* A) \ (δt .* f)
end

# --- Assemble a Duffing oscillator with a classic chaotic double-well parameter set ---
function duffing_model(; δ = 0.3, α = -1.0, β = 1.0)
    params = (δ = δ, α = α, β = β)
    return ODEModel(params, duffing_rhs, (p, x, u) -> duffing_jacobian(p, x), duffing_predict)
end
