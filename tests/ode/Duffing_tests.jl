# Validation tests for the Duffing oscillator model. The linearly-implicit trapezoidal
# `predict` map is checked against the linear system it solves, at a state held fixed by a
# matching forcing, and for its second-order accuracy under step refinement.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Test, LinearAlgebra

# --- Include repo files relative to the root (avoids repeated joinpath calls) ---
const ROOT = normpath(@__DIR__, "..", "..")
use(parts...) = Base.include(Main, joinpath(ROOT, parts...))
use("models", "ode", "ode.jl")

# --- March the model from x0 over ts with constant forcing u ---
function rollout(model, u, ts, x0)
    X = Matrix{Float64}(undef, length(x0), length(ts)); X[:, 1] = x0
    for n in 2:length(ts)
        X[:, n] = model.predict(model.params, X[:, n-1], u, ts[n] - ts[n-1])
    end
    return X
end

@testset "Duffing oscillator" begin
    u = [0.0, 0.3]

    @testset "linearly-implicit trapezoidal step solves its linear system" begin
        model = duffing_model()
        p = model.params
        x, δt = [0.5, -0.2], 0.01
        x_mid = x .+ (δt / 2) .* model.rhs(p, x, u)     # explicit midpoint predictor
        A = duffing_jacobian(p, x_mid)
        x1 = model.predict(p, x, u, δt)
        @test (I - (δt / 2) .* A) * (x1 .- x) ≈ δt .* model.rhs(p, x, u)
    end

    @testset "a state is held fixed by a matching forcing" begin
        model = duffing_model()
        p = model.params
        x = [0.5, -0.2]
        u_eq = -model.rhs(p, x, zeros(2))          # cancels the vector field at x
        @test model.predict(p, x, u_eq, 0.05) ≈ x
    end

    @testset "predict is second-order accurate under step refinement" begin
        model = duffing_model()
        x0, t_end = [0.5, 0.5], 0.5              # off-equilibrium start (not a well bottom)
        ref = rollout(model, zeros(2), 0.0:1e-4:t_end, x0)[:, end]
        e1 = norm(rollout(model, zeros(2), 0.0:2e-3:t_end, x0)[:, end] - ref)
        e2 = norm(rollout(model, zeros(2), 0.0:1e-3:t_end, x0)[:, end] - ref)
        @test 3.5 < e1 / e2 < 4.5                  # halving δt cuts the error ~4×
    end
end
