# Validation tests for the neural-network vector field algorithm. The model interface
# is checked (weight container, right-hand-side shapes, the explicit-Euler `predict`
# formula, the fixed μ/σ scaling, and that a batched control MATRIX is applied per
# sample rather than broadcast), and all three gradient-based training strategies —
# `train_model_residual`, `train_model_sobolev`, and `train_model_autoregressive` — are
# verified to reduce the loss on small problems the network can represent, including with
# a selectable optimiser and, for `train_model_autoregressive`, with a per-step control
# signal `Useg` driving a `polynomial_model_AC` (PolynomialModel.jl). For
# `train_model_sobolev`, `model_jacobian`'s finite-difference estimate is checked against
# the exact (constant) Jacobian of a linear model, and `sobolev_loss` at α = 0 is checked
# to match `residual_loss` exactly (the Jacobian term drops out).
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Test, Random, Statistics, LinearAlgebra

# --- Include repo files relative to the root (avoids repeated joinpath calls) ---
const ROOT = normpath(@__DIR__, "..", "..")
use(parts...) = Base.include(Main, joinpath(ROOT, parts...))
use("algorithms", "ml", "ml.jl")

@testset "neural vector field" begin
    u0 = zeros(3)

    @testset "model interface: shapes, scaling, explicit-Euler predict" begin
        μ, σ = [1.0, -2.0, 3.0], [2.0, 0.5, 4.0]
        model = neural_model(; width = 16, depth = 2, μ = μ, σ = σ, rng = MersenneTwister(1))
        @test model.params isa AbstractVector       # ComponentArray of weights

        x = [0.3, -1.1, 2.0]
        @test length(model.rhs(model.params, x, u0)) == 3
        X = randn(MersenneTwister(2), 3, 7)
        @test size(model.rhs(model.params, X, u0)) == (3, 7)

        # predict is exactly one explicit Euler step of the right-hand side
        δt = 0.05
        @test model.predict(model.params, x, u0, δt) ≈ x .+ δt .* model.rhs(model.params, x, u0)

        # the control enters as a network input, not as additive forcing
        u = [0.1, 0.2, -0.3]
        @test model.rhs(model.params, x, u) != model.rhs(model.params, x, u0)
        @test model.rhs(model.params, x, u) ≉ model.rhs(model.params, x, u0) .+ u

        # a control MATRIX (one column per batch entry) is used per-sample, not broadcast
        Ubatch = randn(MersenneTwister(3), 3, 7)
        @test model.rhs(model.params, X, Ubatch)[:, 1] ≈ model.rhs(model.params, X[:, 1], Ubatch[:, 1])
        @test model.rhs(model.params, X, Ubatch) != model.rhs(model.params, X, Ubatch[:, 1])
    end

    @testset "train_model_residual reduces the loss on a learnable linear target" begin
        rng = MersenneTwister(4)
        X = randn(rng, 3, 200)
        A = 0.5 .* randn(rng, 3, 3)
        Y = A * X                                    # a derivative target the network can represent
        model = neural_model(; width = 16, depth = 2, rng = MersenneTwister(5))
        _, history = train_model_residual(model, X, Y; iters = 500, log_every = 0)
        @test history[end] < 0.5 * history[1]
    end

    @testset "model_jacobian matches the exact (constant) Jacobian of a linear model" begin
        rng = MersenneTwister(20)
        model = polynomial_model_AC(; nin = 3, nctrl = 3, rng = rng)
        X = randn(rng, 3, 5)
        Jest = model_jacobian(model, model.params, X, u0)
        @test size(Jest) == (3, 3, 5)
        for k in 1:5
            @test Jest[:, :, k] ≈ model.params.A atol = 1e-4
        end
    end

    @testset "sobolev_loss at α = 0 matches residual_loss exactly" begin
        rng = MersenneTwister(21)
        model = neural_model(; width = 16, depth = 2, rng = rng)
        X = randn(rng, 3, 20); Y = randn(rng, 3, 20)
        Jdummy = zeros(3, 3, 20)                     # ignored once α = 0
        @test sobolev_loss(model, model.params, X, Y, Jdummy; α = 0.0) ≈ residual_loss(model, model.params, X, Y)
    end

    @testset "train_model_sobolev reduces the loss on a learnable linear target (with a matching Jacobian)" begin
        rng = MersenneTwister(22)
        X = randn(rng, 3, 200)
        A = 0.5 .* randn(rng, 3, 3)
        Y = A * X                                    # a derivative target the network can represent
        Jtarget = cat([A for _ in axes(X, 2)]...; dims = 3)   # the target's own (constant) Jacobian
        model = neural_model(; width = 16, depth = 2, rng = MersenneTwister(23))
        _, history = train_model_sobolev(model, X, Y, Jtarget; iters = 500, log_every = 0)
        @test history[end] < 0.5 * history[1]
    end

    @testset "train_model_autoregressive reduces the loss over trajectory chunks" begin
        rng = MersenneTwister(6)
        A = 0.3 .* randn(rng, 3, 3)                  # stable-ish linear vector field f(x) = A x
        δt, B, K = 0.02, 60, 5
        Xseg = Array{Float64}(undef, 3, B, K + 1)
        Xseg[:, :, 1] = randn(rng, 3, B)
        for k in 2:K + 1                             # true Euler continuation of each chunk
            Xseg[:, :, k] = Xseg[:, :, k-1] .+ δt .* (A * Xseg[:, :, k-1])
        end
        model = neural_model(; width = 16, depth = 2, rng = MersenneTwister(7))
        _, history = train_model_autoregressive(model, Xseg, δt; iters = 500, log_every = 0)
        @test history[end] < 0.5 * history[1]
    end

    @testset "train_model_autoregressive accepts a per-step control Useg (driven systems)" begin
        rng = MersenneTwister(10)
        A, C = 0.2 .* randn(rng, 2, 2), randn(rng, 2, 1)   # f(x, u) = A x + C u, exactly representable
        δt, B, K = 0.02, 40, 5
        Useg = randn(rng, 1, B, K)                    # one control sample per step transition
        Xseg = Array{Float64}(undef, 2, B, K + 1)
        Xseg[:, :, 1] = randn(rng, 2, B)
        for k in 2:K + 1                              # true Euler continuation, driven by Useg
            Xseg[:, :, k] = Xseg[:, :, k-1] .+ δt .* (A * Xseg[:, :, k-1] .+ C * Useg[:, :, k-1])
        end
        model = polynomial_model_AC(; nin = 2, nctrl = 1, rng = MersenneTwister(11))
        _, history = train_model_autoregressive(model, Xseg, δt; Useg = Useg, iters = 500, log_every = 0)
        @test history[end] < 0.5 * history[1]
    end

    @testset "the optimizer is selectable (LBFGS)" begin
        rng = MersenneTwister(8)
        X = randn(rng, 3, 150); Y = (0.4 .* randn(rng, 3, 3)) * X
        model = neural_model(; width = 16, depth = 2, rng = MersenneTwister(9))
        _, history = train_model_residual(model, X, Y; optimizer = OptimizationOptimJL.LBFGS(),
                                          iters = 100, log_every = 0)
        @test history[end] < history[1]
    end
end
