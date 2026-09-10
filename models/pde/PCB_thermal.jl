# Transient 2D heat conduction on a PCB geometry as a `PDEModel`.
#

# The PCB thermal model is a transient linear heat conduction problem with varying heat conductivity
# as well as heating h and cooling c loads restricted to certain locations:
#
#             dT/dt = div(κ(x,y) grad T) + h(t,x,y) + c(t,x,y),
#
# which, discretised Finite Elements or Finite Volumes, gives the semi-discrete system
#
#             M dT/dt = -K ⋅ T + h(t) ⋅ s_h  + c(t) ⋅ s_c.
# Dirichlet data (default T = 0 on the left and right edges) is carried in `params.bc`. The
# constrained implicit-step matrix is factorized once per step size (cached in `params.cache`),
# and only the right-hand side is constrained anew on each step.
#
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using SparseArrays, LinearAlgebra

# --- Semi-discrete field  M dT/dt = -K T + load(u)  and its state Jacobian ---
pcb_load(p, u)        = p.s_c .* u[1] .+ p.s_h .* u[2]
pcb_rhs(p, T, u)      = -p.K * T .+ pcb_load(p, u)
pcb_jacobian(p, T, u) = -p.K

# --- One implicit-Euler step: constrained matrix factorized once per δt, only the rhs constrained per step ---
function pcb_predict(p, T, u, δt)
    A0, F = get!(p.cache, δt) do                     # constrain and factorize A once per step size δt
        A0 = p.M .+ δt .* p.K
        A, _ = impose_dirichlet(A0, p.g; p.bc...)
        (A0, lu(A))
    end
    b, _ = impose_dirichlet(p.M * T .+ δt .* pcb_load(p, u), A0, p.g; p.bc...)
    return F \ b
end

# --- Assemble a PCB thermal model from its conductivity and load-distribution fields ---
function pcb_thermal_model(g::Grid, field_κ, field_s_c, field_s_h;
                           assemble = assemble_fem_matrices, load_distribution = assemble_fem_load_distribution,
                           bc = (left = 0.0, right = 0.0))
    M, K = assemble(g, field_κ)
    s_c  = load_distribution(g, field_s_c)
    s_h  = load_distribution(g, field_s_h)
    params = (g = g, M = M, K = K, s_c = s_c, s_h = s_h, bc = bc, cache = Dict{Float64,Any}())
    return PDEModel(params, pcb_load, pcb_rhs, pcb_jacobian, pcb_predict)
end
