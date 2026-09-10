# Benchmark of linear solvers for the stationary FEM heat conduction problem.
#
# Assembles the finite element system  K ⋅ T = s_c ⋅ c + s_h ⋅ h  on the PCB geometry
# (Dirichlet T = 0 on the left and right edges) and times a couple of different
# linear solvers on the very same system: direct factorisations (sparse
# backslash reference, LU / UMFPACK, Cholesky / CHOLMOD, QR / SPQR) and Krylov.jl
# iterative solvers (cg, gmres) each combined with a few preconditioners (none,
# Jacobi, incomplete LU, algebraic multigrid). For the direct factorisations the
# factorisation and the re-usable triangular solve are timed separately, and each
# preconditioner build is timed as its setup cost, since real-time twins
# typically factor / set up once and re-solve many times. Prints a timing table
# and saves a bar chart of the total solve times.
#
# TODO: Add a study over different mesh sizes (ndof) and plot the scaling of the total solve times.
# TODO: Add a study over different preconditioner parameters (e.g. ILU drop tolerance) 
# TODO: Add a convergence study over different mesh sizes
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using SparseArrays, LinearAlgebra, Plots, Images
using Krylov, IncompleteLU, AlgebraicMultigrid

println("\nBenchmarking direct and iterative linear solvers on the stationary PCB heat conduction FEM system...\n")

# --- Axis tick labels rounded to 3 significant digits on all plots ---
default(formatter = y -> y isa Number ? string(round(y, sigdigits = 3)) : string(y))

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

# --- Assemble the FEM system and impose Dirichlet T = 0 on left / right ---
c, h = -1.0, 1.0
_, K0 = assemble_fem_matrices(g, field_κ)
s_c = assemble_fem_load_distribution(g, field_s_c)
s_h = assemble_fem_load_distribution(g, field_s_h)
b0 = s_c .* c .+ s_h .* h
K, _ = impose_dirichlet(K0, g; left = 0.0, right = 0.0)
b, _ = impose_dirichlet(b0, K0, g; left = 0.0, right = 0.0)

# --- Solver tolerances, timing helper, and shared reference solution ---
tol, maxit = 1e-8, 2000
timeit(f; samples = 5) = (f(); minimum(@elapsed f() for _ in 1:samples))  # fastest of a few runs
results = NamedTuple[]

# --- Reference: sparse backslash (single-shot factor + solve) ---
t_solve = timeit(() -> K \ b)
T_ref   = K \ b
push!(results, (name = "backslash (reference)", setup = NaN, solve = t_solve,
                total = t_solve, err = 0.0, iters = 0))

# Direct solver: time the factorisation and the re-usable triangular solve separately.
function bench_direct!(name, fact)
    t_fact  = timeit(() -> fact(K))
    F       = fact(K)
    t_solve = timeit(() -> F \ b)
    T       = F \ b
    push!(results, (name = name, 
                    setup = t_fact, solve = t_solve, total = t_fact + t_solve, 
                    err = norm(T - T_ref) / norm(T_ref), iters = 0))
end

# --- Direct factorisations, one solver after another ---
bench_direct!("LU (UMFPACK)",       lu)
bench_direct!("Cholesky (CHOLMOD)", A -> cholesky(Symmetric(A)))
bench_direct!("QR (SPQR)",          qr)

# Iterative solver: time the preconditioner build (setup) and the Krylov solve separately.
function bench_iterative!(name, build_pc, solve)
    t_setup  = build_pc === nothing ? 0.0 : timeit(build_pc)
    P        = build_pc === nothing ? nothing : build_pc()
    t_solve  = timeit(() -> solve(P))
    T, stats = solve(P)
    push!(results, (name = name, 
                    setup = t_setup, solve = t_solve, total = t_setup + t_solve, 
                    err = norm(T - T_ref) / norm(T_ref), iters = stats.niter))
end

# --- Conjugate gradient (SPD) with different preconditioners ---
bench_iterative!("cg",          nothing, P -> cg(K, b; atol = 0.0, rtol = tol, itmax = maxit))
bench_iterative!("cg + Jacobi", () -> Diagonal(1 ./ diag(K)),  P -> cg(K, b; M = P, atol = 0.0, rtol = tol, itmax = maxit))
bench_iterative!("cg + AMG",    () -> aspreconditioner(ruge_stuben(K)),
                 P -> cg(K, b; M = P, ldiv = true, atol = 0.0, rtol = tol, itmax = maxit))

# --- GMRES with different preconditioners ---
bench_iterative!("gmres",          nothing, P -> gmres(K, b; atol = 0.0, rtol = tol, itmax = maxit))
bench_iterative!("gmres + Jacobi", () -> Diagonal(1 ./ diag(K)),  P -> gmres(K, b; M = P, atol = 0.0, rtol = tol, itmax = maxit))
bench_iterative!("gmres + ILU",    () -> ilu(K; τ = 0.01),
                 P -> gmres(K, b; M = P, ldiv = true, atol = 0.0, rtol = tol, itmax = maxit))

# --- Print the timing table ---
fmt_row(name, a, b, c, d, e) = rpad(name, 22) * " " * lpad(a, 10) * " " * lpad(b, 10) * " " * lpad(c, 10) * " " * lpad(d, 10) * " " * lpad(e, 7)
ms(x) = isnan(x) ? "-" : string(round(1e3x, digits = 2))

println("FEM linear solver benchmark  (ndof = $(size(K, 1)), nnz(K) = $(nnz(K)))\n")
println(fmt_row("solver", "setup/ms", "solve/ms", "total/ms", "rel.err", "iters"))
for r in results
    println(fmt_row(r.name, ms(r.setup), ms(r.solve), string(round(1e3 * r.total, digits = 2)),
                     string(round(r.err, sigdigits = 2)), r.iters == 0 ? "-" : string(r.iters)))
end



# ============================== VISUALIZATION ===============================

# --- Save a bar chart of the total solve times ---
plt = bar([r.name for r in results], [1e3 * r.total for r in results];
          legend = false, ylabel = "total solve time (ms)", xrotation = 25,
          title = "FEM linear solver benchmark (ndof = $(size(K, 1)))",
          size = (900, 500))
resultdir = joinpath(ROOT, "results", "pde"); mkpath(resultdir)
savefig(plt, joinpath(resultdir, "PCB_benchmark.png"))
