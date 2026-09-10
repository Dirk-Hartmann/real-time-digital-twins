# Compare a Krylov-subspace (moment-matching) ROM against the full transient FE
# simulation of the PCB heat problem. The grid and one reference trajectory are taken
# from the PCB training ensemble (data/PCB_training.bson); the full operators M, K and
# the input matrix B = [s_c s_h] are assembled from the geometry images, and
# homogeneous Dirichlet T = 0 is imposed on the left/right edges as in the reference
# (which also makes K invertible for the Krylov subspace). A Krylov ROM of dimension r
# is built from the operators alone (it is input-independent) and then driven by the
# same random signals c(t), h(t) recorded for that trajectory. The script reports the
# reduction error and visualises the full vs reduced fields, their difference, and the
# error over time.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using SparseArrays, LinearAlgebra, Plots, Images, BSON

println("\nComparing a Krylov-subspace ROM against the full transient PCB FE simulation...")

# --- Axis tick labels rounded to 3 significant digits on all plots ---
default(formatter = y -> y isa Number ? string(round(y, sigdigits = 3)) : string(y))

# --- Include repo files relative to the root (avoids repeated joinpath calls) ---
const ROOT = normpath(@__DIR__, "..", "..")
use(parts...) = Base.include(Main, joinpath(ROOT, parts...))
use("util", "util.jl")
use("models", "pde", "pde.jl")
use("algorithms", "rom", "rom.jl")

# --- Load one reference trajectory from the PCB training ensemble, build the grid from it ---
datadir = joinpath(ROOT, "data")
geomdir = joinpath(ROOT, "models", "data")
data = BSON.load(joinpath(datadir, "PCB_training.bson"))
sim = 1                                                # pick one simulation from the ensemble as reference
ts = data[:t]
c, h = data[:u][1, :, sim], data[:u][2, :, sim]
u = [[c[n], h[n]] for n in eachindex(ts)]              # stacked input [c; h] per step
g = Grid(data[:nelx], data[:nely]; δx = data[:δx])
Xref = data[:T][:, :, sim]                             # full-order reference snapshot matrix

# --- Assemble the FE system and impose Dirichlet T = 0 on the left/right edges ---
field_κ   = Float64.(Gray.(load(joinpath(geomdir, "PCB_geometry_k.png"))))
field_s_c = Float64.(Gray.(load(joinpath(geomdir, "PCB_geometry_c.png"))))
field_s_h = Float64.(Gray.(load(joinpath(geomdir, "PCB_geometry_h.png"))))
M, K = assemble_fem_matrices(g, field_κ)
B = hcat(assemble_fem_load_distribution(g, field_s_c), assemble_fem_load_distribution(g, field_s_h))
ndof = g.nx * g.ny
M, fixed = impose_dirichlet(M, g; left = 0.0, right = 0.0)
K, _     = impose_dirichlet(K, g; left = 0.0, right = 0.0)
B[fixed, :] .= 0.0

# --- Build the Krylov ROM and roll it over ts (encode, predict, decode) ---
r = 20
rom = rom_krylov(M, K, B; r = r)
Z = Matrix{Float64}(undef, r, length(ts))
Z[:, 1] = rom.encode(rom.params, zeros(ndof))
for n in 2:length(ts)
    Z[:, n] = rom.predict(rom.params, Z[:, n-1], u[n], ts[n] - ts[n-1])
    n % 100 == 0 && println("  step $n / $(length(ts))")
end
Xrom = rom.decode(rom.params, Z)

# --- Reduction error over time (relative 2-norm) ---
relerr = [norm(Xref[:, n] - Xrom[:, n]) / (norm(Xref[:, n]) + eps()) for n in eachindex(ts)]
println("Krylov ROM  r = $r   mean rel. error = ",
        round(sum(relerr) / length(relerr), sigdigits = 4),
        "   max = ", round(maximum(relerr), sigdigits = 4))



# ============================== VISUALIZATION ===============================

# --- Reshape nodal vectors to (ny x nx) grids at the final time ---
Tfull = tomat(g, Xref[:, end]); Trom = tomat(g, Xrom[:, end])
clims = extrema(Tfull)

# --- Visualise ROM solution, its difference to the full solution, error, and inputs ---
hm(z, title; kw...) = heatmap(z; title, yflip = true, aspect_ratio = 1,
                              axis = false, ticks = false, kw...)
plt = plot(
    hm(Trom,         "T Krylov ROM (t = $(round(ts[end], digits = 1)) s, r = $r)"; color = :thermal, clims = clims),
    hm(Tfull - Trom, "difference to full solution";                               color = :balance),
    plot(ts, relerr; xlabel = "t [s]", ylabel = "rel. error",
         title = "ROM error over time", legend = false, lw = 2),
    plot(ts, [c h]; xlabel = "t [s]", title = "input signals",
         label = ["c(t)" "h(t)"], lw = 2),
    layout = (2, 2), size = (1100, 900),
)

# --- Store the comparison figure in results/rom ---
resultdir = joinpath(ROOT, "results", "rom"); mkpath(resultdir)
savefig(plt, joinpath(resultdir, "PCB_KrylovROM.png"))
