# Validation tests for the 2D PCB heat conduction discretisations and the
# `pcb_thermal_model` PDEModel: FE/FV assembly, Dirichlet imposition, and the
# load / rhs / jacobian / predict maps of the assembled model.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Test, SparseArrays, LinearAlgebra

# --- Include repo files relative to the root (avoids repeated joinpath calls) ---
const ROOT = normpath(@__DIR__, "..", "..")
use(parts...) = Base.include(Main, joinpath(ROOT, parts...))

use("util", "StructuredGrid.jl")
use("models", "pde", "pde.jl")

@testset "heat conduction assembly" begin
    nelx, nely = 12, 10
    g = Grid(nelx, nely)
    field_κ = fill(2.0, nely, nelx)

    M_fe, K_fe = assemble_fem_matrices(g, field_κ)
    M_fv, K_fv = assemble_fvm_matrices(g, field_κ)
    s_h_fe = assemble_fem_load_distribution(g, ones(nely, nelx))
    s_h_fv = assemble_fvm_load_distribution(g, ones(nely, nelx))

    @testset "symmetry and null space" begin
        for (M, K) in ((M_fe, K_fe), (M_fv, K_fv))
            @test issymmetric(Matrix(K))
            @test issymmetric(Matrix(M))
            @test norm(K * ones(g.nx * g.ny)) < 1e-10   # constant field -> zero Neumann residual
        end
        # K is positive semidefinite (conduction operator)
        @test minimum(eigvals(Symmetric(Matrix(K_fe)))) > -1e-10
        @test minimum(eigvals(Symmetric(Matrix(K_fv)))) > -1e-10
    end

    @testset "conductivity scales the operator linearly" begin
        _, K1 = assemble_fem_matrices(g, fill(1.0, nely, nelx))
        _, K3 = assemble_fem_matrices(g, fill(3.0, nely, nelx))
        @test Matrix(K3) ≈ 3 .* Matrix(K1)
    end

    @testset "mass and load-distribution integrals" begin
        area = (nelx * g.δx) * (nely * g.δx)
        @test sum(M_fe) ≈ area
        @test sum(M_fv) ≈ area
        @test sum(s_h_fe) ≈ area          # unit field integrated over the domain
        @test sum(s_h_fv) ≈ area
        @test s_h_fe ≈ s_h_fv             # FE lumped load matches FV load
        @test all(s_h_fv .>= 0)
        # FV mass is diagonal (lumped control volumes)
        @test M_fv ≈ spdiagm(0 => diag(M_fv))
    end

    @testset "load distribution is linear in the field" begin
        f = [Float64(elx + ely) for ely in 1:nely, elx in 1:nelx]
        @test assemble_fem_load_distribution(g, 2 .* f) ≈ 2 .* assemble_fem_load_distribution(g, f)
    end

    @testset "patch test (linear field is reproduced exactly)" begin
        # kappa constant, no source, Dirichlet boundary = exact linear field T = x
        xnode(ix) = (ix - 1) * g.δx
        left   = zeros(g.ny)
        right  = fill(xnode(g.nx), g.ny)
        topbot = [xnode(ix) for ix in 1:g.nx]
        exact  = [xnode(ix) for iy in 1:g.ny, ix in 1:g.nx]

        for assemble in (assemble_fem_matrices, assemble_fvm_matrices)
            _, K = assemble(g, field_κ)
            b = zeros(g.nx * g.ny)
            A, _ = impose_dirichlet(K, g; left = left, right = right,
                                    top = topbot, bottom = topbot)
            b, _ = impose_dirichlet(b, K, g; left = left, right = right,
                                    top = topbot, bottom = topbot)
            T = reshape(A \ b, g.ny, g.nx)
            @test maximum(abs, T .- exact) < 1e-10
        end
    end

    @testset "FE and FV agree on a smooth problem" begin
        # manufactured smooth source, homogeneous Dirichlet, refined mesh
        n = 60
        gg = Grid(n, n)
        f = [sin(pi * (elx - 0.5) / n) * sin(pi * (ely - 0.5) / n) for ely in 1:n, elx in 1:n]
        field_κ = ones(n, n)
        sols = map(((assemble_fem_matrices, assemble_fem_load_distribution),
                    (assemble_fvm_matrices, assemble_fvm_load_distribution))) do (matrices, load)
            _, K = matrices(gg, field_κ)
            b = load(gg, f)
            A, _ = impose_dirichlet(K, gg; left = 0.0, right = 0.0, top = 0.0, bottom = 0.0)
            b, _ = impose_dirichlet(b, K, gg; left = 0.0, right = 0.0, top = 0.0, bottom = 0.0)
            A \ b
        end
        @test norm(sols[1] - sols[2]) / norm(sols[1]) < 5e-2
    end
end

@testset "Dirichlet imposition" begin
    g = Grid(6, 5)
    _, K = assemble_fem_matrices(g, ones(g.nely, g.nelx))

    @testset "matrix map: zeroed rows/cols and unit diagonal" begin
        A, fixed = impose_dirichlet(K, g; left = 0.0, right = 0.0)
        @test sort(fixed) == sort(vcat(left_nodes(g), right_nodes(g)))
        allbut(d) = setdiff(1:size(A, 1), d)
        for dof in fixed
            @test A[dof, dof] == 1.0
            @test norm(A[dof, allbut(dof)]) == 0.0        # row cleared off-diagonal
            @test norm(A[allbut(dof), dof]) == 0.0        # column cleared off-diagonal
        end
        @test issymmetric(Matrix(A))                      # elimination preserves symmetry
        @test K == assemble_fem_matrices(g, ones(g.nely, g.nelx))[2]  # original untouched (copy)
    end

    @testset "rhs map writes prescribed values" begin
        b0 = randn(g.nx * g.ny)
        b, _ = impose_dirichlet(b0, K, g; left = 1.5, right = -0.5)
        @test b[left_nodes(g)] == fill(1.5, g.ny)
        @test b[right_nodes(g)] == fill(-0.5, g.ny)
    end

    @testset "solving reproduces the prescribed boundary values" begin
        A, _ = impose_dirichlet(K, g; left = 2.0, right = 3.0)
        b, _ = impose_dirichlet(zeros(g.nx * g.ny), K, g; left = 2.0, right = 3.0)
        T = A \ b
        @test T[left_nodes(g)] ≈ fill(2.0, g.ny)
        @test T[right_nodes(g)] ≈ fill(3.0, g.ny)
        @test all(2.0 - 1e-9 .<= T .<= 3.0 + 1e-9)        # discrete maximum principle
    end

    @testset "scalar and vector specs agree; shared corners take the last side" begin
        fixed_s, vals_s = dirichlet_dofs(g; left = 4.0)
        fixed_v, vals_v = dirichlet_dofs(g; left = fill(4.0, g.ny))
        @test fixed_s == fixed_v && vals_s == vals_v
        # top is specified after left, so the shared top-left corner takes the top value
        d = Dict(zip(dirichlet_dofs(g; left = 1.0, top = 9.0)...))
        @test d[node_id(g, 1, 1)] == 9.0
    end
end

@testset "PCB thermal model (PDEModel)" begin
    g = Grid(16, 12)
    field_κ   = fill(1.5, g.nely, g.nelx)
    field_s_c = fill(0.5, g.nely, g.nelx)
    field_s_h = fill(1.0, g.nely, g.nelx)
    model = pcb_thermal_model(g, field_κ, field_s_c, field_s_h)
    p = model.params
    T = randn(g.nx * g.ny)
    u = (-1.3, 0.7)                                        # (cooling c, heating h)

    @testset "load, rhs and jacobian assemble the semi-discrete field" begin
        @test model.load(p, u) ≈ p.s_c .* u[1] .+ p.s_h .* u[2]
        @test model.rhs(p, T, u) ≈ -p.K * T .+ model.load(p, u)
        @test model.jacobian(p, T, u) == -p.K
    end

    @testset "predict solves the constrained implicit-Euler system" begin
        δt = 0.05
        A0 = p.M .+ δt .* p.K
        A, _ = impose_dirichlet(A0, p.g; p.bc...)
        b, _ = impose_dirichlet(p.M * T .+ δt .* model.load(p, u), A0, p.g; p.bc...)
        @test model.predict(p, T, u, δt) ≈ A \ b
    end

    @testset "predict respects the Dirichlet boundary" begin
        Tn = model.predict(p, T, u, 0.05)
        @test Tn[left_nodes(g)] ≈ zeros(g.ny) atol = 1e-10
        @test Tn[right_nodes(g)] ≈ zeros(g.ny) atol = 1e-10
    end

    @testset "the step matrix is factorized once per step size (cache)" begin
        m = pcb_thermal_model(g, field_κ, field_s_c, field_s_h)
        @test isempty(m.params.cache)
        m.predict(m.params, T, u, 0.05)
        @test length(m.params.cache) == 1
        m.predict(m.params, T, u, 0.05)                   # same δt reuses the factorization
        @test length(m.params.cache) == 1
        m.predict(m.params, T, u, 0.10)                   # new δt adds an entry
        @test length(m.params.cache) == 2
    end

    @testset "unforced homogeneous problem dissipates energy" begin
        m = pcb_thermal_model(g, field_κ, field_s_c, field_s_h)
        T0 = randn(g.nx * g.ny)
        T1 = m.predict(m.params, T0, (0.0, 0.0), 0.05)
        @test norm(T1) < norm(T0)                         # implicit Euler is contractive
    end

    @testset "steady heating drives a fixed point of predict" begin
        m = pcb_thermal_model(g, field_κ, field_s_c, field_s_h)
        u_ss = (0.0, 1.0)
        # steady state: K T = load with Dirichlet imposed
        A, _ = impose_dirichlet(m.params.K, g; m.params.bc...)
        b, _ = impose_dirichlet(m.load(m.params, u_ss), m.params.K, g; m.params.bc...)
        T_ss = A \ b
        @test m.predict(m.params, T_ss, u_ss, 0.05) ≈ T_ss   # steady state is invariant
    end

    @testset "FE and FV models can both be assembled" begin
        fem = pcb_thermal_model(g, field_κ, field_s_c, field_s_h;
                                assemble = assemble_fem_matrices,
                                load_distribution = assemble_fem_load_distribution)
        fvm = pcb_thermal_model(g, field_κ, field_s_c, field_s_h;
                                assemble = assemble_fvm_matrices,
                                load_distribution = assemble_fvm_load_distribution)
        @test fem.params.s_c ≈ fvm.params.s_c             # lumped loads coincide
        @test size(fem.params.K) == size(fvm.params.K)
        @test issymmetric(Matrix(fvm.params.K))
    end
end
