# Convolutional autoencoder reduction: a nonlinear, translation-equivariant
# alternative to the dense POD-autoencoder of AE-POD-Reduction.jl. Rather than
# compressing a POD coordinate vector, the encoder e_θ acts on the full field
# reshaped to its (ny × nx) grid: a single strided convolution downsamples it to a
# small (fy × fx) feature map (kept as the only full-resolution convolution, for
# speed), optionally refined there by cheap same-resolution convolutions, then
# flattened to a dense bottleneck. The decoder d_θ mirrors this — dense expansion,
# same-resolution refinement, then one upsample back to the grid resolution
# (bilinear interpolation, which avoids the checkerboard artefacts of transposed
# convolutions) followed by the single full-resolution convolution to one output
# channel,
#     encode(θ, x) : Rⁿ → Rˡ,     decode(θ, z) : Rˡ → Rⁿ,     x ≈ d_θ(e_θ(x)).
# It shares the `DimReduction` interface (encode/decode, no `predict`) and folds in
# fixed input statistics μ, σ as constants exactly like the POD-autoencoder. An optional
# `dev` keyword places the parameters on another device (e.g. `Lux.gpu_device()`); the
# caller is then responsible for moving the training data (and matching μ, σ) to the
# same device before calling `encode`/`decode`. `train_CNN_autoencoder` fits θ by
# gradient descent on the reconstruction loss (no time stepping), optionally tracking
# a held-out validation set alongside; it mirrors `train_POD_autoencoder`
# (AE-POD-Reduction.jl) but kept self-contained here.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using Lux, ComponentArrays, Random, Optimization, OptimizationOptimisers, Zygote, Statistics

# --- Build a convolutional encoder e_θ / decoder d_θ pair wrapped in the reduction interface ---
function CNN_autoencoder(; g::Grid, latent, channels = 4, depth = 2, stride = 8, act = tanh,
                          μ = 0.0, σ = 1.0, rng = Random.default_rng(), dev = identity)
    ny, nx = g.ny, g.nx

    # --- Encoder: one strided convolution to a small (fy × fx) map, refined there cheaply ---
    down = Conv((5, 5), 1 => channels, act; stride = stride, pad = SamePad())
    pb, sb = Lux.setup(rng, down)
    fshape = size(down(zeros(Float32, ny, nx, 1, 1), pb, sb)[1])[1:2]   # probe the downsampled size
    refine_enc = ntuple(_ -> Conv((3, 3), channels => channels, act; pad = SamePad()), depth - 1)
    flat = prod(fshape) * channels
    encoder = Chain(down, refine_enc..., FlattenLayer(), Dense(flat => latent))

    # --- Decoder: mirror the refinement, then one upsample and convolution back to the grid ---
    refine_dec = ntuple(_ -> Conv((3, 3), channels => channels, act; pad = SamePad()), depth - 1)
    decoder = Chain(Dense(latent => flat, act), ReshapeLayer((fshape..., channels)), refine_dec...,
                     Upsample(:bilinear; size = (ny, nx)), Conv((5, 5), channels => 1; pad = SamePad()),
                     FlattenLayer())

    pe, se = Lux.setup(rng, encoder)
    pd, sd = Lux.setup(rng, decoder)
    if dev === identity                                    # Float32 already suits a GPU device
        pe, pd = Lux.f64(pe), Lux.f64(pd)                   # match the Float64 field data on CPU
    end
    params = dev(ComponentArray(enc = pe, dec = pd))

    # --- Column-wise maps: a single state (vector) or a whole n×T series (matrix) in one call ---
    tobatch(x::AbstractVector) = reshape(x, :, 1)
    tobatch(x::AbstractMatrix) = x
    toimg(x) = reshape((tobatch(x) .- μ) ./ σ, ny, nx, 1, :)
    function encode(θ, x)
        z = encoder(toimg(x), θ.enc, se)[1]
        return x isa AbstractVector ? vec(z) : z
    end
    function decode(θ, z)
        x̂ = σ .* decoder(tobatch(z), θ.dec, sd)[1] .+ μ
        return z isa AbstractVector ? vec(x̂) : x̂
    end
    return DimReduction(params, encode, decode)
end

# --- Full-state reconstruction loss (mean squared error, weighted by σ, plus an L2 penalty) ---
CNN_autoencoder_loss(ae, θ, X; σ = ones(size(X, 1)), λ = 1e-4) =
    mean(abs2, (ae.decode(θ, ae.encode(θ, X)) .- X) ./ σ) + λ * sum(abs2, θ)

# --- Fit encoder/decoder weights to reconstruct the snapshots X; returns θ and loss log.
#     Passing a held-out Xval tracks its loss alongside every `val_every` iterations
#     (returned as a third output, logged every iteration by default); the convolutional
#     forward pass is expensive enough that `val_every` can be raised to reduce overhead ---
function train_CNN_autoencoder(ae, X; Xval = nothing, σ = ones(size(X, 1)), λ = 1e-4, θ0 = ae.params,
                                optimizer = OptimizationOptimisers.Adam(1e-3), iters = 2000,
                                log_every = 200, val_every = 1)
    history = Float64[]
    history_val = Float64[]
    cb = (state, l) -> (push!(history, l);
        Xval !== nothing && length(history) % val_every == 0 &&
            push!(history_val, CNN_autoencoder_loss(ae, state.u, Xval; σ = σ, λ = λ));
        log_every > 0 && length(history) % log_every == 0 &&
            println("  iter $(length(history)) / $iters   loss $(round(l, sigdigits = 4))");
        false)
    optf = OptimizationFunction((θ, _) -> CNN_autoencoder_loss(ae, θ, X; σ = σ, λ = λ), Optimization.AutoZygote())
    res  = solve(OptimizationProblem(optf, θ0), optimizer; maxiters = iters, callback = cb)
    return Xval === nothing ? (res.u, history) : (res.u, history, history_val)
end
