# Finite element assembly for 2D heat conduction on a structured Q1 (bilinear,
# 4-node quadrilateral) grid.
#
# Assembles M and T the semi-discrete system  
#             M dT/dt = -K ⋅ T + load(t)
# using a Finite Element Method (FEM) with Q1 elements (bilinear, 4-node quadrilaterals),
# where K is the symmetric positive-semidefinite conductivity matrix from the governing PDE,
# which is dT/dt = div(κ grad T) + load(t)
#
# Element matrices have been calculated by means of symbolic integration on a
# unit cell and are scaled by the cell-wise conductivity κ[ely, elx]. 
# Zero-Neumann boundaries are the default (no boundary term is added). Other boundary
# conditions can be imposed by modifying the system after assembly (see impose_dirichlet).
#
# The nodal load-distribution vectors weights a cell-wise field by the Q1 shape-function integrals.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

using SparseArrays

# --- Q1 element matrices on a square cell of side δx ---
# The Laplacian conductivity matrix is independent of δx and κ (scaled later).
element_conductivity() = (1 / 6) .* [ 4.0  -1.0  -2.0  -1.0
                                     -1.0   4.0  -1.0  -2.0
                                     -2.0  -1.0   4.0  -1.0
                                     -1.0  -2.0  -1.0   4.0 ]

element_mass(δx) = (δx^2 / 36) .* [ 4.0  2.0  1.0  2.0
                                    2.0  4.0  2.0  1.0
                                    1.0  2.0  4.0  2.0
                                    2.0  1.0  2.0  4.0 ]

# --- Assemble the mass and conductivity matrices M, K with a finite element method ---
function assemble_fem_matrices(g::Grid, field_κ)
    ndof = g.nx * g.ny
    Ke = element_conductivity()
    Me = element_mass(g.δx)

    I = Int[]; J = Int[]; vec_K = Float64[]; vec_M = Float64[]
    for elx in 1:g.nelx, ely in 1:g.nely                 # loop over elements
        edof = element_nodes(g, elx, ely)
        for a in 1:4, b in 1:4                           # scatter local into global (sparse assembly)
            push!(I, edof[a]); push!(J, edof[b])
            push!(vec_K, field_κ[ely, elx] * Ke[a, b])   # conductivity matrix K
            push!(vec_M, Me[a, b])                       # mass matrix M
        end
    end
    K = sparse(I, J, vec_K, ndof, ndof)
    M = sparse(I, J, vec_M, ndof, ndof)
    return M, K
end

# --- Assemble a nodal load-distribution vector from a cell-wise field (Q1 finite element) ---
function assemble_fem_load_distribution(g::Grid, field)
    fe = fill(g.δx^2 / 4, 4)   # integral of each shape function over the cell
    s = zeros(g.nx * g.ny)
    for elx in 1:g.nelx, ely in 1:g.nely
        edof = element_nodes(g, elx, ely)
        for a in 1:4
            s[edof[a]] += field[ely, elx] * fe[a]
        end
    end
    return s
end
