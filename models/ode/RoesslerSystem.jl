# Rössler system, a minimal chaotic ordinary differential equation
#             dx₁/dt = -x₂ - x₃
#             dx₂/dt = x₁ + a x₂
#             dx₃/dt = b + x₃ (x₁ - c)
# written for the state vector x = [x₁, x₂, x₃] as  dx/dt = f(x) + u.
# For the classic parameters a = 0.2, b = 0.2, c = 5.7 the flow
# folds onto a single-scroll strange attractor.
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

# --- Definition of the Rössler system ---
roessler_rhs(p, x, u) = [ -x[2] - x[3],
                          x[1] + p.a * x[2],
                          p.b + x[3] * (x[1] - p.c) ] .+ u

roessler_jacobian(p, x) = [ 0.0   -1.0        -1.0
                            1.0    p.a         0.0
                            x[3]   0.0    x[1] - p.c ]

# --- One linearly-implicit trapezoidal step (A frozen at the midpoint predictor) ---
function roessler_predict(p, x, u, δt)
    f = roessler_rhs(p, x, u)
    x_mid = x .+ (δt / 2) .* f
    A = roessler_jacobian(p, x_mid)
    return x .+ (I - (δt / 2) .* A) \ (δt .* f)
end

# --- Assemble a Rössler model with the classic parameters as defaults ---
function roessler_model(; a = 0.2, b = 0.2, c = 5.7)
    params = (a = a, b = b, c = c)
    return ODEModel(params, roessler_rhs, (p, x, u) -> roessler_jacobian(p, x), roessler_predict)
end
