# Krylov-subspace (moment-matching) reduced-order model for the LTI system
#             M dx/dt = -K x + B u(t).
# The transfer function H(s) = Cᵀ (sM + K)⁻¹ B is approximated by matching its
# leading moments about an expansion point s₀: the projection basis spans the block
# Krylov subspace
#     colspan{ R, A R, A² R, … },   A = (K + s₀M)⁻¹ M,   R = (K + s₀M)⁻¹ B,
# so the reduced transfer function reproduces the first moments of the full one and
# the ROM is accurate independently of the specific input signal. An orthonormal
# basis is built block-by-block with a block Arnoldi iteration (modified
# Gram–Schmidt with reorthogonalisation) and grown until it reaches the requested
# dimension r; the reduced operators then follow from a Galerkin projection.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using LinearAlgebra

# --- Block Arnoldi construction of a moment-matching basis and reduced model ---
function rom_krylov(M, K, B; r::Integer, s0::Real = 0.0)
    fact = lu(K .+ s0 .* M)
    W = Matrix(fact \ Matrix(B))                       # starting block R = (K + s₀M)⁻¹ B
    V = zeros(size(W, 1), 0)
    while size(V, 2) < r
        W .-= V * (V' * W); W .-= V * (V' * W)         # orthogonalise (twice) against V
        Q = Matrix(qr(W).Q * Matrix(1.0I, size(W)...)) # thin orthonormal basis of the block
        take = min(r - size(V, 2), size(Q, 2))
        V = hcat(V, Q[:, 1:take])
        W = Matrix(fact \ (M * V[:, end-take+1:end]))  # next Krylov block A·(new columns)
    end

    # --- Reduced operators and projection maps  (x ≈ V z) ---
    params = (V = V, M = V' * (M * V), K = V' * (K * V), B = V' * Matrix(B))
    encode(p, x) = p.V' * x
    decode(p, z) = p.V * z
    predict(p, z, u, δt) = (p.M .+ δt .* p.K) \ (p.M * z .+ δt .* (p.B * u))
    return ReducedModel(params, encode, decode, predict)
end
