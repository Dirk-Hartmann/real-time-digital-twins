# Common interface for the ordinary-differential-equation models in this directory.
# An `ODEModel` bundles the physical `params` with two maps that both take those
# parameters explicitly as their first argument, so every ODE model  shares the exact
# same signature:
#   rhs(params, x, u)         state x and forcing u   ->  dx/dt = f(x) + u,
#   jacobian(params, x, u)    state x and forcing u   ->  ∂f/∂x, the Jacobian of the field,
#   predict(params, x, u, δt) advance the state x by one step of size δt.
# The state vector is x and the forcing/control is u (both vary with time). Each model in this
# directory constructs its own `ODEModel`.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

struct ODEModel
    params    # physical parameters of the model
    rhs       # (params, x(t), u(t))      -> dx(t)/dt      the vector field with forcing
    jacobian  # (params, x(t), u(t))      -> ∂f/∂x         Jacobian of the vector field
    predict   # (params, x(t), u(t), δt)  -> x(t+δt)       one step of size δt
end
