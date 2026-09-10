# Validation tests for the SLM moving-source thermal model: the Gaussian
# load-distribution field, and the `slm_thermal_model` PDEModel maps
# (load / rhs / jacobian / predict) including the folded dissipation β M.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Test, SparseArrays, LinearAlgebra

# --- Include repo files relative to the root (avoids repeated joinpath calls) ---
const ROOT = normpath(@__DIR__, "..", "..")
use(parts...) = Base.include(Main, joinpath(ROOT, parts...))

use("util", "StructuredGrid.jl")
use("models", "pde", "pde.jl")

# --- Node coordinate of a linear dof index (column-major, iy fastest) ---
node_coord(g, idx) = (((idx - 1) ÷ g.ny) * g.δx, ((idx - 1) % g.ny) * g.δx)

@testset "SLM Gaussian load distribution" begin
    g = Grid(100, 100)

    @testset "Gaussian field integrates to the prescribed power" begin
        s = assemble_fem_load_distribution(g, gaussian_distribution_field(g, 0.5, 0.5; power = 2.0, width = 0.05))
        @test sum(s) ≈ 2.0 rtol = 1e-2
        @test all(s .>= 0)
    end

    @testset "power scales the field linearly" begin
        f1 = gaussian_distribution_field(g, 0.5, 0.5; power = 1.0, width = 0.05)
        f3 = gaussian_distribution_field(g, 0.5, 0.5; power = 3.0, width = 0.05)
        @test f3 ≈ 3 .* f1
    end

    @testset "load peaks at the spot centre" begin
        xc, yc = 0.3, 0.7
        s = assemble_fem_load_distribution(g, gaussian_distribution_field(g, xc, yc; width = 0.04))
        x, y = node_coord(g, argmax(s))
        @test hypot(x - xc, y - yc) < 2 * g.δx
    end
end

@testset "SLM thermal model (PDEModel)" begin
    g = Grid(60, 60)
    field_κ = fill(1.0, g.nely, g.nelx)
    β, power, width = 8.0, 1.0, 0.05
    model = slm_thermal_model(g, field_κ; power = power, width = width, β = β)
    p = model.params
    T = randn(g.nx * g.ny)
    u = (0.4, 0.6)

    @testset "dissipation β M is folded into the operator" begin
        M, K = assemble_fem_matrices(g, field_κ)
        @test p.K ≈ K .+ β .* M
        @test model.jacobian(p, T, u) == -p.K
    end

    @testset "load re-assembles the Gaussian at the spot centre" begin
        @test model.load(p, u) ≈
              assemble_fem_load_distribution(g, gaussian_distribution_field(g, u[1], u[2]; power = power, width = width))
        @test model.rhs(p, T, u) ≈ -p.K * T .+ model.load(p, u)
    end

    @testset "predict solves the implicit-Euler system (zero-Neumann, no BC)" begin
        δt = 0.005
        A = p.M .+ δt .* p.K
        @test model.predict(p, T, u, δt) ≈ A \ (p.M * T .+ δt .* model.load(p, u))
    end

    @testset "unforced field decays toward the ambient temperature" begin
        cold = slm_thermal_model(g, field_κ; power = 0.0, width = width, β = β)
        T0 = randn(g.nx * g.ny)
        T1 = cold.predict(cold.params, T0, u, 0.005)      # zero power -> zero load
        @test norm(T1) < norm(T0)                          # dissipation contracts the state
    end
end

@testset "SLM hot spot follows the laser" begin
    g = Grid(100, 100)
    model = slm_thermal_model(g, fill(1.0, g.nely, g.nelx); power = 1.0, width = 0.03, β = 10.0)
    δt = 0.005
    laser(t) = (0.15 + 0.6 * t, 0.5)

    U = zeros(g.nx * g.ny)
    for n in 1:100
        U = model.predict(model.params, U, laser(n * δt), δt)
    end
    xc, yc = laser(100 * δt)
    x, y = node_coord(g, argmax(U))
    @test hypot(x - xc, y - yc) < 0.05                     # peak temperature trails the spot
    @test all(U .>= -1e-12)                                # nonnegative heating, dissipation to 0
end
