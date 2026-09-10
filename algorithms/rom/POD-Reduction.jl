# Proper Orthogonal Decomposition (POD) as a plain dimensionality reduction — the
# linear counterpart of the autoencoder, and the basis part of `rom_pod` without the
# Galerkin projection or any time stepping. Given a snapshot matrix S whose columns
# are states x(t_k), POD picks the subspace that optimally captures the snapshot
# energy: the leading left singular vectors of the thin SVD
#     S = U Σ Wᵀ   truncated to   V = U[:, 1:r].
# The retained energy fraction is Σᵢ₌₁ʳ σᵢ² / Σᵢ σᵢ². With V orthonormal the reduction
# is the orthogonal projection and its transpose,
#     encode(x) = Vᵀ x,     decode(z) = V z,
# so `pod_reduction` returns a `DimReduction` with the same interface as the learned
# autoencoder, letting the two be compared mode for mode.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using LinearAlgebra

# --- POD basis (leading left singular vectors of the snapshots) as a reduction ---
function pod_reduction(S; r::Integer)
    V = svd(Matrix(S)).U[:, 1:r]
    encode(p, x) = p.V' * x
    decode(p, z) = p.V * z
    return DimReduction((V = V,), encode, decode)
end
