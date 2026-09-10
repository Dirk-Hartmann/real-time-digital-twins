# Validation tests for the reduced-order models (Krylov moment matching and POD).
# A small heat problem with two localised sources and homogeneous Dirichlet edges
# is reduced; the reduced trajectories are compared against a full-order implicit-
# Euler reference on the free dofs.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Test, SparseArrays, LinearAlgebra

# --- Include repo files relative to the root (avoids repeated joinpath calls) ---
const ROOT = normpath(@__DIR__, "..", "..")
use(parts...) = Base.include(Main, joinpath(ROOT, parts...))
use("util", "util.jl")
use("models", "pde", "pde.jl")
use("algorithms", "rom", "rom.jl")

# --- Full-order implicit-Euler reference and reduced-model rollout ---
function integrate(M, K, B, u, ts, x0)
    δt = ts[2] - ts[1]
    fact = lu(M .+ δt .* K)
    X = Matrix{Float64}(undef, length(x0), length(ts)); X[:, 1] = x0
    for n in 2:length(ts)
        X[:, n] = fact \ (M * X[:, n-1] .+ δt .* (B * u[n]))
    end
    return X
end
function rollout(rom, u, ts, x0)
    z = rom.encode(rom.params, x0)
    Z = Matrix{Float64}(undef, length(z), length(ts)); Z[:, 1] = z
    for n in 2:length(ts)
        Z[:, n] = rom.predict(rom.params, Z[:, n-1], u[n], ts[n] - ts[n-1])
    end
    return rom.decode(rom.params, Z)
end

@testset "reduced-order models" begin
    # --- Small full problem: two Gaussian sources, Dirichlet on left/right ---
    nelx, nely = 24, 20
    g = Grid(nelx, nely)
    field_κ   = fill(1.0, nely, nelx)
    field_s_c = [exp(-((elx - 6)^2  + (ely - 6)^2)  / 8) for ely in 1:nely, elx in 1:nelx]
    field_s_h = [exp(-((elx - 18)^2 + (ely - 14)^2) / 8) for ely in 1:nely, elx in 1:nelx]
    M, K = assemble_fem_matrices(g, field_κ)
    B = hcat(assemble_fem_load_distribution(g, field_s_c), assemble_fem_load_distribution(g, field_s_h))
    fixed = sort(unique([left_nodes(g); right_nodes(g)]))
    free  = setdiff(1:g.nx * g.ny, fixed)
    M_ff, K_ff, B_f = M[free, free], K[free, free], B[free, :]

    # --- Full transient reference (snapshot matrix, columns are states) ---
    ts = 0.0:0.1:5.0
    u  = [[sin(0.5t), cos(0.3t)] for t in ts]
    x0 = zeros(length(free))
    S  = integrate(M_ff, K_ff, B_f, u, ts, x0)
    relmax(Xrom) = maximum(norm(S[:, n] - Xrom[:, n]) / (norm(S[:, n]) + eps()) for n in eachindex(ts))

    @testset "reduced dimensions and orthonormal basis" begin
        p = rom_pod(S, M_ff, K_ff, B_f; r = 8).params
        @test size(p.M) == (8, 8)
        @test size(p.B) == (8, 2)
        @test p.V' * p.V ≈ I
        @test p.K ≈ p.K'                           # projection preserves symmetry
    end

    @testset "encode / decode handle a time series" begin
        rom = rom_pod(S, M_ff, K_ff, B_f; r = 12)
        Z = rom.encode(rom.params, S)              # encode the whole trajectory at once
        @test size(Z) == (12, length(ts))
        @test rom.decode(rom.params, rom.encode(rom.params, rom.params.V)) ≈ rom.params.V
    end

    @testset "predict advances one step of the requested size" begin
        rom = rom_pod(S, M_ff, K_ff, B_f; r = 12)
        p = rom.params
        z = rom.encode(p, x0)
        δt = 0.07                                  # an arbitrary step size
        z1 = rom.predict(p, z, u[2], δt)
        @test (p.M .+ δt .* p.K) * z1 ≈ p.M * z .+ δt .* (p.B * u[2])
    end

    @testset "Krylov ROM matches the full transient" begin
        rom = rom_krylov(M_ff, K_ff, B_f; r = 16)
        @test rom.params.V' * rom.params.V ≈ I
        @test relmax(rollout(rom, u, ts, x0)) < 5e-2
    end

    @testset "POD-Galerkin ROM reproduces its snapshots" begin
        rom = rom_pod(S, M_ff, K_ff, B_f; r = 10)
        @test rom.params.V' * rom.params.V ≈ I
        @test relmax(rollout(rom, u, ts, x0)) < 1e-2
    end
end
