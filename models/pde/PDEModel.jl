# Common interface for the semi-discrete partial-differential-equation models in this
# directory. A `PDEModel` bundles the physical `params` (grid, assembled operators, and
# load-distribution patterns) with four maps that all take those parameters explicitly as their first
# argument, so every PDE model shares the exact same signature:
#   load(params, u)           forcing u               ->  the nodal load vector,
#   rhs(params, x, u)         state x and forcing u   ->  M ⋅ dx/dt = -K ⋅ x + load(params, u),
#   jacobian(params, x, u)    state x and forcing u   ->  ∂(rhs)/∂x = -K,
#   predict(params, x, u, δt) advance the state x by one implicit step of size δt.
# The state vector is x and the forcing/control is u (both vary with
# time). Each model in this directory constructs its own `PDEModel`.
#
# TODO: Is there a benefit of having the 'load' explicitly in the struct?
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

struct PDEModel
    params    # grid, assembled operators (M, K), and load-distribution patterns of the model
    load      # (params, u(t))            -> nodal load vector           forcing assembled from u
    rhs       # (params, x(t), u(t))      -> M ⋅ dx/dt = -K ⋅ x + load    semi-discrete field
    jacobian  # (params, x(t), u(t))      -> ∂(rhs)/∂T = -K              Jacobian of the field
    predict   # (params, x(t), u(t), δt)  -> x(t+δt)                     one implicit step
end
