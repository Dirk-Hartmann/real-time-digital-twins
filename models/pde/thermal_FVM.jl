# Finite volume assembly for 2D heat conduction on a structured grid.
#
# Assembles the semi-discrete system  
#              M dT/dt = -K ⋅ T + load(t)
# using a Finite Volume Method (FVM) with a 5-point stencil,
# where K is the symmetric positive-semidefinite conductivity matrix from
# the governing PDE, which is dT/dt = div(κ grad T) + load(t)
#
# Face conductivities are obtained from the cell-wise field field_κ[ely, elx] by
# averaging the cells adjacent to each edge; in 2D the mesh size cancels so each edge
# contributes a conductance equal to the face conductivity. Mass and load-distributions are lumped
# onto the nodal control volumes (quarter-cell areas), which makes the load-distribution vectors 
# identical to the FEM ones. Missing edges at the boundary encode the default 
# zero-Neumann condition. Other boundary conditions can be imposed by modifying the
# system after assembly (see impose_dirichlet).
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using SparseArrays

# --- Collect information for FVM assembly on a structured grid ---
function add_edge!(I, J, V, P, Q, κ)
    push!(I, P); push!(J, P); push!(V,  κ)
    push!(I, Q); push!(J, Q); push!(V,  κ)
    push!(I, P); push!(J, Q); push!(V, -κ)
    push!(I, Q); push!(J, P); push!(V, -κ)
end

# --- Assemble the mass and conductivity matrices M, K with a finite volume method ---
function assemble_fvm_matrices(g::Grid, field_κ)
    ndof = g.nx * g.ny
    I = Int[]; J = Int[]; vec_K = Float64[]

    # --- x-edges: face between horizontally adjacent nodes (cells above / below) ---
    for iy in 1:g.ny, ix in 1:g.nx - 1
        κ = 0.0; n = 0
        if iy - 1 >= 1;      κ += field_κ[iy - 1, ix]; n += 1 end
        if iy     <= g.nely; κ += field_κ[iy,     ix]; n += 1 end
        add_edge!(I, J, vec_K, node_id(g, ix, iy), node_id(g, ix + 1, iy), κ / n)
    end

    # --- y-edges: face between vertically adjacent nodes (cells left / right) ---
    for ix in 1:g.nx, iy in 1:g.ny - 1
        κ = 0.0; n = 0
        if ix - 1 >= 1;      κ += field_κ[iy, ix - 1]; n += 1 end
        if ix     <= g.nelx; κ += field_κ[iy, ix];     n += 1 end
        add_edge!(I, J, vec_K, node_id(g, ix, iy), node_id(g, ix, iy + 1), κ / n)
    end

    K = sparse(I, J, vec_K, ndof, ndof)   # conductivity matrix K

    # --- lumped mass from quarter-cell areas ---
    area = zeros(ndof); A4 = g.δx^2 / 4
    for elx in 1:g.nelx, ely in 1:g.nely
        for (ix, iy) in ((elx, ely), (elx + 1, ely), (elx, ely + 1), (elx + 1, ely + 1))
            area[node_id(g, ix, iy)] += A4
        end
    end
    M = spdiagm(0 => area)
    return M, K
end

# --- Assemble a nodal load-distribution vector from a cell-wise field (lumped finite volume) ---
function assemble_fvm_load_distribution(g::Grid, field)
    s = zeros(g.nx * g.ny); A4 = g.δx^2 / 4
    for elx in 1:g.nelx, ely in 1:g.nely
        for (ix, iy) in ((elx, ely), (elx + 1, ely), (elx, ely + 1), (elx + 1, ely + 1))
            s[node_id(g, ix, iy)] += field[ely, elx] * A4
        end
    end
    return s
end
