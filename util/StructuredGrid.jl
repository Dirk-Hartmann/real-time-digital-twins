# Structured quadrilateral grid for 2D problems.
# Node numbering is column-major.
# The same grid is used for both FEM and FVM discretisations, so that the unknowns are shared.
#
#        ix ->                     column-major node ids (nelx=nely=2):
#     (1,1)   (2,1)   (3,1)
#  iy   o-------o-------o           col1 col2 col3
#   |   | e11   | e21   |            1    4    7   <- iy=1 (top)
#   v   o-------o-------o            2    5    8   <- iy=2
#       | e12   | e22   |            3    6    9   <- iy=3 (bottom)
#       o-------o-------o
#
# Element (elx,ely) corner nodes in local order [TL, TR, BR, BL]:
#   n1 = (elx-1)*ny + ely,  n2 = elx*ny + ely,  edof = (n1, n2, n2+1, n1+1)
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

# --- Grid structure and initialization ---
struct Grid
    nelx::Int
    nely::Int
    nx::Int      # nodes in x = nelx + 1
    ny::Int      # nodes in y = nely + 1
    δx::Float64  # cell edge length (Δx = Δy = δx)
end

Grid(nelx::Integer, nely::Integer; δx::Real = 1.0 / nelx) =
    Grid(nelx, nely, nelx + 1, nely + 1, float(δx))

# --- Node indexing and element connectivity ---
node_id(g::Grid, ix::Integer, iy::Integer) = (ix - 1) * g.ny + iy

function element_nodes(g::Grid, elx::Integer, ely::Integer)
    n1 = (elx - 1) * g.ny + ely
    n2 =  elx      * g.ny + ely
    return (n1, n2, n2 + 1, n1 + 1)   # TL, TR, BR, BL
end

# --- Cell-centre coordinates on the unit domain ---
cell_center(g::Grid, elx::Integer, ely::Integer) = ((elx - 0.5) * g.δx, (ely - 0.5) * g.δx)

# --- Reshape a node vector to a grid matrix ---
tomat(g::Grid, T) = reshape(T, g.ny, g.nx)

# --- Boundary node lists (ordered along each side) ---
left_nodes(g::Grid)   = [node_id(g, 1,    iy) for iy in 1:g.ny]
right_nodes(g::Grid)  = [node_id(g, g.nx, iy) for iy in 1:g.ny]
top_nodes(g::Grid)    = [node_id(g, ix, 1)    for ix in 1:g.nx]
bottom_nodes(g::Grid) = [node_id(g, ix, g.ny) for ix in 1:g.nx]
