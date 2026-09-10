# Study the Proper Orthogonal Decomposition (POD) of the PCB heat problem on its
# own, without the Galerkin projection or any time integration. The 10 trajectories
# of the PCB training ensemble (data/PCB_training.bson) are split 80/20 into training
# and validation TRAJECTORIES (so no snapshot of a validation trajectory leaks into
# training) and each split is flattened into one snapshot matrix; the POD basis
# V = U[:, 1:r] is built from the training snapshots alone (leading left singular
# vectors), and its ability to represent unseen states is measured by the
# reconstruction (encode-then-decode) error V Vᵀx on both sets. Because U spans only
# the training snapshots, the validation error saturates at the energy of the
# component lying outside that span, whereas the training error keeps decaying with
# r. The script visualises the POD eigenvalues σᵢ², the energy omitted by the first
# k modes, and the mean reconstruction error of both sets as a function of the
# number of retained modes.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using LinearAlgebra, Plots, BSON

println("\nStudying the reconstruction quality of a POD basis for the PCB heat problem...")

# --- Axis tick labels rounded to 3 significant digits on all plots ---
default(formatter = y -> y isa Number ? string(round(y, sigdigits = 3)) : string(y))

# --- Include repo files relative to the root (avoids repeated joinpath calls) ---
const ROOT = normpath(@__DIR__, "..", "..")
use(parts...) = Base.include(Main, joinpath(ROOT, parts...))
use("util", "util.jl")

# --- Load the PCB training ensemble ---
data = BSON.load(joinpath(ROOT, "data", "PCB_training.bson"))
ts = data[:t]

# --- Split whole trajectories into training and validation sets (80/20), then flatten each ---
ntraj = size(data[:T], 3)
train, validation, _ = split_trajectories(ntraj; val_frac = 0.2, test_frac = 0.0)
Xtrain = reshape(data[:T][:, :, train], size(data[:T], 1), :)
Xval   = reshape(data[:T][:, :, validation], size(data[:T], 1), :)

# --- POD of the training snapshots: basis U, eigenvalues λ = σ², omitted energy ---
F = svd(Xtrain)
U, σ = F.U, F.S
λ = σ .^ 2
omitted = max.(1 .- cumsum(λ) ./ sum(λ), eps())

# --- Mean relative encode/decode error V Vᵀx over the columns of X, per mode count ---
function recon_errors(U, X, rs)
    A = U' * X                                          # projection coordinates V'x (all modes)
    oos2 = vec(sum(abs2, X .- U * A, dims = 1))         # energy outside the training span
    captured = cumsum(A .^ 2, dims = 1)                 # energy in the first r modes, per column
    colnorm = map(norm, eachcol(X)) .+ eps()
    [sum(sqrt.(oos2 .+ (captured[end, :] .- captured[r, :])) ./ colnorm) / length(colnorm) for r in rs]
end

# --- Reconstruction error on both sets versus the number of retained modes ---
rs = 1:min(40, length(σ))
err_train = recon_errors(U, Xtrain, rs)
err_val   = recon_errors(U, Xval, rs)
println("POD study: $(size(Xtrain, 2)) train / $(size(Xval, 2)) validation snapshots, $(size(Xtrain, 1)) dofs")
for r in (5, 10, 15)
    println("    r = $(lpad(r, 2))   omitted energy = $(round(omitted[r], sigdigits = 3))   train err = $(round(err_train[r], sigdigits = 3))   val err = $(round(err_val[r], sigdigits = 3))")
end



# ============================== VISUALIZATION ===============================

# --- Visualise POD eigenvalues, omitted energy, and reconstruction error ---
plt = plot(
    plot(λ; yscale = :log10, xlabel = "mode i", ylabel = "eigenvalue σᵢ²",
         title = "POD eigenvalues", legend = false, lw = 2, marker = :circle, ms = 3),
    plot(omitted; yscale = :log10, xlabel = "modes k", ylabel = "omitted energy",
         title = "POD truncation error", legend = false, lw = 2, marker = :circle, ms = 3),
    plot(rs, [err_train err_val]; yscale = :log10, xlabel = "retained modes r",
         ylabel = "rel. reconstruction error", title = "encode/decode error",
         label = ["train" "validation"], lw = 2, marker = :circle, ms = 3),
    layout = (1, 3), size = (1500, 450),
)

# --- Store the POD analysis figure in results/rom ---
resultdir = joinpath(ROOT, "results", "rom"); mkpath(resultdir)
savefig(plt, joinpath(resultdir, "PCB_POD.png"))
