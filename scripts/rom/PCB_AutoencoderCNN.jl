# Study a learned convolutional autoencoder reduction of the PCB heat problem and
# compare it against the linear POD of PCB_POD.jl. Unlike the POD-autoencoder of
# PCB_AutoencoderPOD.jl — which compresses the leading POD coordinates with dense
# layers — the encoder/decoder here (AE-CNN-Reduction.jl) act directly on the
# full temperature field reshaped to its (ny × nx) grid, exploiting spatial locality
# with convolutions instead of a change of coordinates. The 10 trajectories of
# data/PCB_training.bson are split 80/20 into training and validation TRAJECTORIES
# (so no snapshot of a validation trajectory leaks into training), each flattened
# into one snapshot matrix. The network is fitted on the GPU at a single latent size
# ℓ = 8 (no resolution sweep), using only a quarter of the training snapshots — every
# 4th time step, taken uniformly from every training trajectory (not by dropping whole
# trajectories) — to keep training time in check. The relative full-state
# reconstruction error is measured on the full training and validation sets against
# the POD baseline V Vᵀx at the same latent size. The script visualises the training
# loss and a reconstructed field.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using LinearAlgebra, Statistics, Random, Plots, BSON

println("\nTraining a convolutional autoencoder reduction of the PCB heat problem...")

# --- Work around a Windows cuDNN artifact whose engine DLL isn't on the default search path ---
for (root, _, files) in walkdir(joinpath(DEPOT_PATH[1], "artifacts"))
    if "cudnn_engines_tensor_ir64_9.dll" in files
        ENV["PATH"] = root * ";" * ENV["PATH"]
        break
    end
end
using LuxCUDA

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

# --- Quarter the training snapshots for the (slow) CNN training step: every 4th time
#     step, taken uniformly from every training trajectory (not by dropping trajectories) ---
ntime = size(data[:T], 2)
Xtrain_cnn = reshape(data[:T][:, 1:4:ntime, train], g.nx * g.ny, :)

# --- POD baseline basis and scalar statistics of the training field ---
p = 40
U = svd(Xtrain).U[:, 1:p]
μ, σ = mean(Xtrain), std(Xtrain)

# --- Relative full-state reconstruction error of a decoded snapshot set (skips the
#     degenerate T = 0 initial-condition snapshots, one per simulation) ---
relerr(Xtrue, X̂) = mean(norm(Xtrue[:, j] - X̂[:, j]) / norm(Xtrue[:, j])
                         for j in axes(Xtrue, 2) if norm(Xtrue[:, j]) > 0)
pod_recon(k, X) = (Uk = U[:, 1:k]; Uk * (Uk' * X))

# --- Train the autoencoder on the GPU at a fixed latent size, using the quartered training snapshots ---
latent = 8
gdev = Lux.gpu_device()
todevice(X) = gdev(Float32.(X))
tohost(X) = Float64.(Array(X))
ae = CNN_autoencoder(; g = g, latent = latent, μ = Float32(μ), σ = Float32(σ),
                     rng = MersenneTwister(0), dev = gdev)
θ, history, history_val = train_CNN_autoencoder(ae, todevice(Xtrain_cnn); Xval = todevice(Xval), σ = Float32(σ),
                                                 λ = 1f-4, iters = 10000, log_every = 200)
ae_recon(X) = tohost(ae.decode(θ, ae.encode(θ, todevice(X))))
ae_train = relerr(Xtrain, ae_recon(Xtrain))
ae_val   = relerr(Xval,   ae_recon(Xval))
pod_val  = relerr(Xval,   pod_recon(latent, Xval))
println("ℓ = $latent   AE train = $(round(ae_train, sigdigits = 3))  val = $(round(ae_val, sigdigits = 3))  |  POD val = $(round(pod_val, sigdigits = 3))")



# ============================== VISUALIZATION ===============================

# --- Reconstruct the final validation-set field with the trained autoencoder ---
Trec = tomat(g, ae_recon(Xval[:, end:end])[:, 1])
Tfull = tomat(g, Xval[:, end])
clims = extrema(Tfull)
hm(z, title; kw...) = heatmap(z; title, yflip = true, aspect_ratio = 1, axis = false, ticks = false, kw...)

# --- Visualise the training loss (train/validation) and a reconstructed field ---
p1 = plot(history; yscale = :log10, xlabel = "iteration", ylabel = "loss",
          title = "training loss", label = "train", lw = 2)
plot!(p1, history_val; yscale = :log10, label = "val", lw = 2)
plt = plot(
    p1,
    hm(Trec,          "T CNN autoencoder (t = $(round(ts[end], digits = 1)) s, ℓ = $latent)"; color = :thermal, clims = clims),
    hm(Tfull - Trec,  "difference to full solution";                                         color = :balance),
    layout = (1, 3), size = (1500, 450),
)

# --- Store the CNN autoencoder analysis figure in results/rom ---
resultdir = joinpath(ROOT, "results", "rom"); mkpath(resultdir)
savefig(plt, joinpath(resultdir, "PCB_AutoencoderCNN.png"))
