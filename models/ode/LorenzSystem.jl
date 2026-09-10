# Lorenz system, the canonical chaotic ordinary differential equation
#             dx₁/dt = σ (x₂ - x₁)
#             dx₂/dt = x₁ (ρ - x₃) - x₂
#             dx₃/dt = x₁ x₂ - β x₃
# written for the state vector x = [x₁, x₂, x₃] as  dx/dt = f(x) + u. For the classic
# parameters σ = 10, ρ = 28, β = 8/3 the flow settles onto the butterfly-shaped
# strange attractor.
#
# The model uses the `ODEModel` interface, i.e., it bundles its `params` with maps that take
# those parameters explicitly as their first argument. Time stepping uses a linearly-implicit
# (Rosenbrock-type) trapezoidal step built from the true Jacobian A(x) = ∂f/∂x,
#             (I - δt/2 A(x*)) ⋅ (xⁿ⁺¹ - xⁿ) = δt (f(xⁿ) + u),
# in which the next state enters only linearly, so each step is a single 3×3 solve
# with no Newton iteration. Freezing A at the explicit midpoint predictor
# x* = xⁿ + δt/2 (f(xⁿ) + u) keeps the scheme second-order accurate.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using LinearAlgebra

# --- Definition of the Lorenz system ---
lorenz_rhs(p, x, u) = [ p.σ * (x[2] - x[1]),
                        x[1] * (p.ρ - x[3]) - x[2],
                        x[1] * x[2] - p.β * x[3] ] .+ u

lorenz_jacobian(p, x) = [ -p.σ        p.σ   0.0
                           p.ρ - x[3] -1.0  -x[1]
                           x[2]        x[1] -p.β ]

# --- One linearly-implicit trapezoidal step (A frozen at the midpoint predictor) ---
function lorenz_predict(p, x, u, δt)
    f = lorenz_rhs(p, x, u)
    x_mid = x .+ (δt / 2) .* f
    A = lorenz_jacobian(p, x_mid)
    return x .+ (I - (δt / 2) .* A) \ (δt .* f)
end

# --- Assemble a Lorenz model with the classic parameters as defaults ---
function lorenz_model(; σ = 10.0, ρ = 28.0, β = 8 / 3)
    params = (σ = σ, ρ = ρ, β = β)
    return ODEModel(params, lorenz_rhs, (p, x, u) -> lorenz_jacobian(p, x), lorenz_predict)
end
