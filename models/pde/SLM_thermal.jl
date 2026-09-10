# Selective laser melting (SLM) thermal model as a `PDEModel`.
#
# The SLM process is a transient linear heat conduction problem with constant conductivity,
# a Gaussian laser spot that travels across the domain, and linear dissipation to an ambient
# temperature of 0:
#
#             dT/dt = div(κ grad T) + q(t,x,y) - β T,
#
# which, discretised with Q1 elements, gives the semi-discrete system
#
#             M dT/dt = -(K + β M) ⋅ T + q(t)⋅q_dist.
#
# The dissipation β M is folded into the model's conductivity operator. 
# The forcing is the moving spot centre u = (x0, y0); the
# load map re-assembles the nodal vector q from a Gaussian field centred there. Every edge
# stays zero-Neumann, so no boundary term is imposed.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using SparseArrays, LinearAlgebra

# --- Cell-wise Gaussian heat-source / load-distribution field centred at (x0, y0) ---
function gaussian_distribution_field(g::Grid, x0, y0; power = 1.0, width = 0.05)
    a = power / (2 * π * width^2)                        # normalised so ∫ q dΩ ≈ power
    field = zeros(g.nely, g.nelx)
    for elx in 1:g.nelx, ely in 1:g.nely
        x, y = cell_center(g, elx, ely)
        field[ely, elx] = a * exp(-((x - x0)^2 + (y - y0)^2) / (2 * width^2))
    end
    return field
end

# --- Semi-discrete field  M dT/dt = -(K + β M) T + q(u)  and its state Jacobian ---
slm_load(p, u)        = assemble_fem_load_distribution(p.g, gaussian_distribution_field(p.g, u[1], u[2]; power = p.power, width = p.width))
slm_rhs(p, T, u)      = -p.K * T .+ slm_load(p, u)
slm_jacobian(p, T, u) = -p.K

# --- One implicit-Euler step (all edges zero-Neumann, no boundary term) ---
function slm_predict(p, T, u, δt)
    A = p.M .+ δt .* p.K
    return A \ (p.M * T .+ δt .* slm_load(p, u))
end

# --- Assemble an SLM thermal model with a moving Gaussian laser and linear dissipation ---
function slm_thermal_model(g::Grid, field_κ; power = 1.0, width = 0.05, β = 10.0)
    M, K = assemble_fem_matrices(g, field_κ)
    params = (g = g, M = M, K = K .+ β .* M, power = power, width = width)
    return PDEModel(params, slm_load, slm_rhs, slm_jacobian, slm_predict)
end
