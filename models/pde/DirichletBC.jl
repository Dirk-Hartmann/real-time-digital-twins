# Impose Dirichlet boundary conditions on a linear system A ⋅ T = b.
#
# Each side (left, right, top, bottom) is either `nothing` (keep the default
# zero-Neumann condition) or a prescribed value given as a scalar or as a vector
# ordered along that side (length ny for left/right, nx for top/bottom). Shared
# corner nodes take the value of the last side that specifies them.
#
# The classic symmetry-preserving elimination is split into two independent maps that
# share the same per-side spec: `impose_dirichlet(A, g; …)` zeroes the constrained
# rows/columns and puts a unit diagonal on the matrix, while
# `impose_dirichlet(b, A, g; …)` moves the known columns of the *original* A to the
# right-hand side and writes the prescribed values into b. Splitting them lets a solver
# constrain and factorize A once and reuse it while only b changes across steps.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

# --- Collect the constrained dofs and their prescribed values from the per-side spec ---
function dirichlet_dofs(g::Grid; left = nothing, right = nothing,
                        top = nothing, bottom = nothing)
    bc = Dict{Int,Float64}()
    for (val, nodes) in ((left, left_nodes(g)), (right, right_nodes(g)),
                         (top, top_nodes(g)), (bottom, bottom_nodes(g)))
        val === nothing && continue
        v = val isa Number ? fill(float(val), length(nodes)) : collect(float.(val))
        @assert length(v) == length(nodes) "boundary value vector length mismatch"
        for (nd, x) in zip(nodes, v)
            bc[nd] = x
        end
    end
    fixed = collect(keys(bc))
    return fixed, [bc[d] for d in fixed]
end

# --- Impose Dirichlet conditions on the matrix: zero constrained rows/cols, unit diagonal ---
function impose_dirichlet(A, g::Grid; kw...)
    A = copy(A)
    fixed, _ = dirichlet_dofs(g; kw...)
    for dof in fixed
        A[dof,   :] .= 0.0
        A[:  , dof] .= 0.0
        A[dof, dof]  = 1.0
    end
    return A, fixed
end

# --- Impose Dirichlet conditions on the rhs: move knowns to RHS via original A, write values ---
function impose_dirichlet(b::AbstractVector, A, g::Grid; kw...)
    b = copy(b)
    fixed, vals = dirichlet_dofs(g; kw...)
    for (dof, val) in zip(fixed, vals)   # move knowns to RHS using original columns
        b .-= A[:, dof] .* val
    end
    b[fixed] .= vals
    return b, fixed
end
