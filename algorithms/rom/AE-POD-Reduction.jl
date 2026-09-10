# POD-autoencoder reduction: a learned, nonlinear autoencoder composed with a linear
# POD pre-projection, so a single `DimReduction` performs both steps. The full state
# is first projected onto the leading p POD coordinates a = Uᵀx (a lossless change of
# coordinates on the span of the training snapshots), then two small multilayer
# perceptrons compress/reconstruct those coordinates,
#     encode(θ, x) = e_θ(Uᵀx) : Rⁿ → Rˡ,     decode(θ, z) = U d_θ(z) : Rˡ → Rⁿ,
# with all weights θ = (enc, dec) fitted jointly by minimising the full-state
# reconstruction error x ≈ decode(θ, encode(θ, x)). Because the POD projection is
# lossless on the training span, a latent space of the same size ℓ can capture more
# of the retained variation than the POD subspace alone. `params` is a NamedTuple
# holding the fixed reduction metadata — `p`, the basis `U`, the per-mode statistics
# `μ, σ` of the training coordinates (folded into the maps as constants, not
# trainable, so the tanh layers see well-scaled inputs while `encode`/`decode` stay
# in physical units), and the bare `encode_pod`/`decode_pod` maps that act on POD
# coordinates directly — together with the trainable weights `enc`, `dec`, so a
# `DimReduction` built by `POD_autoencoder` is fully self-describing. Since U is
# fixed, `train_POD_autoencoder` takes the original trajectories directly: it reads
# `U`, `encode_pod` and `decode_pod` off `ae.params`, projects once up front
# (A = Uᵀx, rather than recomputing — and differentiating — it on every iteration),
# and fits `enc`/`dec` on the projected coordinates; `encode`/`decode` still compose
# U for actual use of the trained reduction.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Lux, ComponentArrays, Random, Optimization, OptimizationOptimisers, Zygote, Statistics, LinearAlgebra

# --- Build a POD projection (basis U from the snapshots X) composed with an encoder
#     e_θ / decoder d_θ pair, wrapped in the reduction interface ---
function POD_autoencoder(X; p, latent, width = 32, depth = 2, act = tanh, rng = Random.default_rng())
    U = svd(X).U[:, 1:p]
    A = U' * X
    μ = vec(mean(A, dims = 2))
    σ = vec(std(A, dims = 2)) .+ eps()

    enc_hidden = ntuple(_ -> Dense(width => width, act), depth - 1)
    dec_hidden = ntuple(_ -> Dense(width => width, act), depth - 1)
    encoder = Chain(Dense(p => width, act), enc_hidden..., Dense(width => latent))
    decoder = Chain(Dense(latent => width, act), dec_hidden..., Dense(width => p))
    pe, se = Lux.setup(rng, encoder)
    pd, sd = Lux.setup(rng, decoder)
    pe, pd = Lux.f64(pe), Lux.f64(pd)      # match the Float64 POD-coordinate data

    # --- Maps on the POD coordinates a = Uᵀx directly (cheap; used to train) ---
    encode_pod(θ, a) = encoder((a .- μ) ./ σ, θ.enc, se)[1]
    decode_pod(θ, z) = σ .* decoder(z, θ.dec, sd)[1] .+ μ

    # --- Full-state maps composing the fixed POD projection U (used to reduce data) ---
    encode(θ, x) = encode_pod(θ, U' * x)
    decode(θ, z) = U * decode_pod(θ, z)

    params = (p = p, U = U, μ = μ, σ = σ, encode_pod = encode_pod, decode_pod = decode_pod,
              enc = pe, dec = pd)   # enc/dec are the trainable part
    return DimReduction(params, encode, decode)
end

# --- Reconstruction loss for a generic encode/decode pair (mean squared error,
#     weighted by σ, plus an L2 penalty) — usable on either the POD coordinates
#     (encode_pod/decode_pod) or the full state (encode/decode) ---
POD_autoencoder_loss(encode, decode, θ, X; σ = ones(size(X, 1)), λ = 1e-4) =
    mean(abs2, (decode(θ, encode(θ, X)) .- X) ./ σ) + λ * sum(abs2, θ)

# --- Fit encoder/decoder weights to reconstruct the original trajectories X (the
#     projection A = Uᵀx onto the POD coordinates, read off ae.params, is done once
#     here rather than recomputed — and differentiated — on every iteration); returns
#     θ and loss log. Passing a held-out Xval tracks its loss alongside every
#     iteration (third output) ---
function train_POD_autoencoder(ae, X; Xval = nothing, σ = ones(ae.params.p), λ = 1e-4,
                                θ0 = ComponentArray(enc = ae.params.enc, dec = ae.params.dec),
                                optimizer = OptimizationOptimisers.Adam(1e-3), iters = 2000,
                                log_every = 200, val_every = 1)
    encode_pod, decode_pod, U = ae.params.encode_pod, ae.params.decode_pod, ae.params.U
    A = U' * X
    Aval = Xval === nothing ? nothing : U' * Xval

    history = Float64[]
    history_val = Float64[]
    cb = (state, l) -> (push!(history, l);
        Aval !== nothing && length(history) % val_every == 0 &&
            push!(history_val, POD_autoencoder_loss(encode_pod, decode_pod, state.u, Aval; σ = σ, λ = λ));
        log_every > 0 && length(history) % log_every == 0 &&
            println("  iter $(length(history)) / $iters   loss $(round(l, sigdigits = 4))");
        false)
    optf = OptimizationFunction((θ, _) -> POD_autoencoder_loss(encode_pod, decode_pod, θ, A; σ = σ, λ = λ),
                                 Optimization.AutoZygote())
    res  = solve(OptimizationProblem(optf, θ0), optimizer; maxiters = iters, callback = cb)
    return Aval === nothing ? (res.u, history) : (res.u, history, history_val)
end
