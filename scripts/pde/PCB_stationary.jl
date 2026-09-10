# Stationary 2D heat conduction on a PCB geometry: FE vs FV comparison.
#
# Solves the steady problem  K T = s_c c + s_h h  with Dirichlet T = 0 on
# the left and right edges (top and bottom stay zero-Neumann), once with the
# finite element and once with the finite volume discretisation, and
# visualises both solutions and their difference.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using SparseArrays, LinearAlgebra, Plots, Images

println("\nSolving the stationary PCB heat conduction problem with FEM and FVM...\n")

# --- Include repo files relative to the root (avoids repeated joinpath calls) ---
const ROOT = normpath(@__DIR__, "..", "..")
use(parts...) = Base.include(Main, joinpath(ROOT, parts...))
use("util", "util.jl")
use("models", "pde", "pde.jl")

# --- Load geometry fields from PNGs ---
datadir = joinpath(ROOT, "models", "data")
field_κ   = Float64.(Gray.(load(joinpath(datadir, "PCB_geometry_k.png"))))
field_s_c = Float64.(Gray.(load(joinpath(datadir, "PCB_geometry_c.png"))))
field_s_h = Float64.(Gray.(load(joinpath(datadir, "PCB_geometry_h.png"))))
nely, nelx = size(field_κ)
g = Grid(nelx, nely)

# --- Stationary drives (heating on, cooling on) ---
c = -1.0
h =  1.0

# --- Steady solve  K T = load  for a given discretisation, with Dirichlet imposed ---
function stationary_solve(model, u)
    p = model.params
    A, _ = impose_dirichlet(p.K, p.g; p.bc...)
    b, _ = impose_dirichlet(model.load(p, u), p.K, p.g; p.bc...)
    return A \ b
end

# --- Solve with the finite element discretisation ---
fem = pcb_thermal_model(g, field_κ, field_s_c, field_s_h;
                        assemble = assemble_fem_matrices, load_distribution = assemble_fem_load_distribution)
T_fe = stationary_solve(fem, (c, h))

# --- Solve with the finite volume discretisation ---
fvm = pcb_thermal_model(g, field_κ, field_s_c, field_s_h;
                        assemble = assemble_fvm_matrices, load_distribution = assemble_fvm_load_distribution)
T_fv = stationary_solve(fvm, (c, h))

# --- Reshape nodal vectors to (ny x nx) grids ---
reldiff = norm(T_fe - T_fv) / norm(T_fe)
println("FE vs FV relative difference (2-norm): ", round(reldiff, sigdigits = 4))
println("max |T_FE - T_FV| = ", round(maximum(abs, T_fe - T_fv), sigdigits = 4))



# ============================== VISUALIZATION ===============================

# --- Visualise inputs, solutions, and their difference ---
hm(z, title; kw...) = heatmap(z; title, yflip = true, aspect_ratio = 1,
                              axis = false, ticks = false, kw...)
plt = plot(
    hm(field_κ,             "conductivity κ"; color = :viridis),
    hm(field_s_c,           "cooling s_c";    color = :blues),
    hm(field_s_h,           "heating s_h";    color = :reds),
    hm(tomat(g, T_fe),      "T (FE)";         color = :thermal),
    hm(tomat(g, T_fv),      "T (FV)";         color = :thermal),
    hm(tomat(g, T_fe-T_fv), "T_FE - T_FV";    color = :balance),
    layout = (2, 3), size = (1200, 800),
)

# --- Store outputs in results/pde ---
resultdir = joinpath(ROOT, "results", "pde"); mkpath(resultdir)
savefig(plt, joinpath(resultdir, "PCB_stationary.png"))
