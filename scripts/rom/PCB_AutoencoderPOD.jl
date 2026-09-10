# Study a learned autoencoder reduction of the PCB heat problem and compare it against
# the linear POD of PCB_POD.jl. The 10 trajectories of the PCB training ensemble
# (data/PCB_training.bson) are split 80/20 into training and validation TRAJECTORIES
# (so no snapshot of a validation trajectory leaks into training), each flattened into
# one snapshot matrix. `POD_autoencoder` (AE-POD-Reduction.jl) folds
# the POD pre-projection onto the leading p = 20 coordinates a = Uᵀx directly into its
# encode/decode maps, a lossless change of coordinates on the training span (a
# "POD-autoencoder"), so a single reduction performs both steps: encode(x) = e_θ(Uᵀx),
# decode(z) = U d_θ(z). `train_POD_autoencoder` takes the original trajectories
# directly and does the projection onto POD coordinates internally (once, not on every
# iteration), reporting the validation loss every iteration. A single latent size
# ℓ = 8 is used (no resolution sweep). The relative full-state reconstruction error
# V Vᵀx (plain POD, POD-Reduction.jl) versus d_θ(e_θ(·)) (autoencoder) is measured on
# both sets. The script visualises the training-loss history and a reconstructed
# field. See PCB_AutoencoderCNN.jl for a convolutional counterpart that acts on the
# field directly instead of its POD coordinates.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using LinearAlgebra, Statistics, Random, Plots, BSON, OptimizationOptimisers

println("\nTraining a POD-autoencoder reduction of the PCB heat problem...")

# --- Axis tick labels rounded to 3 significant digits on all plots ---
default(formatter = y -> y isa Number ? string(round(y, sigdigits = 3)) : string(y))

# --- Include repo files relative to the root (avoids repeated joinpath calls) ---
const ROOT = normpath(@__DIR__, "..", "..")
use(parts...) = Base.include(Main, joinpath(ROOT, parts...))
use("util", "util.jl")
use("algorithms", "rom", "rom.jl")

# --- Load the PCB training ensemble ---
data = BSON.load(joinpath(ROOT, "data", "PCB_training.bson"))
ts = data[:t]
g = Grid(data[:nelx], data[:nely]; δx = data[:δx])

# --- Split whole trajectories into training and validation sets (80/20), then flatten each ---
ntraj = size(data[:T], 3)
train, validation, _ = split_trajectories(ntraj; val_frac = 0.2, test_frac = 0.0)
Xtrain = reshape(data[:T][:, :, train], g.nx * g.ny, :)
Xval   = reshape(data[:T][:, :, validation], g.nx * g.ny, :)

# --- Relative full-state reconstruction error of a decoded snapshot set (skips the
#     degenerate T = 0 initial-condition snapshots, one per simulation) ---
relerr(Xtrue, X̂) = mean(norm(Xtrue[:, j] - X̂[:, j]) / norm(Xtrue[:, j])
                         for j in axes(Xtrue, 2) if norm(Xtrue[:, j]) > 0)

# --- Train the POD-autoencoder on the original trajectories (the projection onto POD
#     coordinates happens once, inside train_POD_autoencoder), with a 10× higher
#     learning rate; the validation loss is reported every iteration. Since a mean
#     squared error averages over p entries per column here instead of n (unlike the
#     full-state loss), σ is scaled by sqrt(n/p) to keep the same effective weight
#     against the L2 penalty λ‖θ‖² as training directly on the full state would ---
p = 20
latent = 8
ae = POD_autoencoder(Xtrain; p = p, latent = latent, rng = MersenneTwister(0))
n = size(Xtrain, 1)
σ = std(Xtrain) * sqrt(n / p)
θ, history, history_val = train_POD_autoencoder(ae, Xtrain; Xval = Xval, σ = σ, iters = 3000,
                                                 log_every = 200, optimizer = OptimizationOptimisers.Adam(1e-2))
ae_train = relerr(Xtrain, ae.decode(θ, ae.encode(θ, Xtrain)))
ae_val   = relerr(Xval,   ae.decode(θ, ae.encode(θ, Xval)))

# --- Plain POD baseline at the same latent size, for comparison ---
podr = pod_reduction(Xtrain; r = latent)
pod_val = relerr(Xval, podr.decode(podr.params, podr.encode(podr.params, Xval)))
println("ℓ = $latent   AE train = $(round(ae_train, sigdigits = 3))  val = $(round(ae_val, sigdigits = 3))  |  POD val = $(round(pod_val, sigdigits = 3))")



# ============================== VISUALIZATION ===============================

# --- Reconstruct the final validation-set field with the trained autoencoder ---
Trec = tomat(g, ae.decode(θ, ae.encode(θ, Xval[:, end:end]))[:, 1])
Tfull = tomat(g, Xval[:, end])
clims = extrema(Tfull)
hm(z, title; kw...) = heatmap(z; title, yflip = true, aspect_ratio = 1, axis = false, ticks = false, kw...)

# --- Visualise the training loss (train/validation) and a reconstructed field ---
p1 = plot(history; yscale = :log10, xlabel = "iteration", ylabel = "loss",
          title = "training loss", label = "train", lw = 2)
plot!(p1, history_val; yscale = :log10, label = "val", lw = 2)
plt = plot(
    p1,
    hm(Trec,          "T autoencoder (t = $(round(ts[end], digits = 1)) s, ℓ = $latent)"; color = :thermal, clims = clims),
    hm(Tfull - Trec,  "difference to full solution";                                     color = :balance),
    layout = (1, 3), size = (1500, 450),
)

# --- Store the autoencoder analysis figure in results/rom ---
resultdir = joinpath(ROOT, "results", "rom"); mkpath(resultdir)
savefig(plt, joinpath(resultdir, "PCB_AutoencoderPOD.png"))
