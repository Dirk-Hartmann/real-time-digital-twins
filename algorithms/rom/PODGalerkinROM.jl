# Proper Orthogonal Decomposition (POD) with Galerkin projection.
# Given a snapshot matrix S whose columns are states x(t_k) sampled from a full
# transient simulation, POD selects the basis that optimally captures the snapshot
# energy: the leading left singular vectors of S. Truncating the thin SVD
#     S = U Σ Wᵀ   to   V = U[:, 1:r]
# yields an orthonormal basis whose retained energy fraction is
# Σᵢ₌₁ʳ σᵢ² / Σᵢ σᵢ². The reduced operators then follow from a Galerkin projection
# onto V. Unlike the Krylov ROM, this basis is tailored to the observed trajectory
# (data-driven), so the snapshots must be representative of the intended operating
# regime.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using LinearAlgebra

# --- POD basis (leading left singular vectors of the snapshots) and reduced model ---
function rom_pod(S, M, K, B; r::Integer)
    V = svd(Matrix(S)).U[:, 1:r]

    # --- Reduced operators and projection maps  (x ≈ V z) ---
    params = (V = V, M = V' * (M * V), K = V' * (K * V), B = V' * Matrix(B))
    encode(p, x) = p.V' * x
    decode(p, z) = p.V * z
    predict(p, z, u, δt) = (p.M .+ δt .* p.K) \ (p.M * z .+ δt .* (p.B * u))
    return ReducedModel(params, encode, decode, predict)
end
