# Common interface for reduced-order models. A `ReducedModel` bundles a set of
# parameters with three maps that all take those parameters explicitly as their
# first argument, so the same interface serves both projection-based ROMs (where
# `params` holds a projection basis and reduced operators) and future learned
# reductions (where `params` holds neural-network weights):
#   encode(params, x)         full state or time series  ->  reduced coordinates,
#   decode(params, z)         reduced coordinates        ->  full state or time series,
#   predict(params, z, u, δt) advance the reduced state by one step of size δt.
# `encode`/`decode` act column-wise, mapping a single state (length-n vector) or a
# whole time series (n×T matrix of column states) in one call; `predict` takes the
# time step δt explicitly, so a model can be marched with varying step sizes. Each
# ROM in this directory constructs its own `ReducedModel`.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

struct ReducedModel
    params    # parameters of the maps (e.g. projection basis + reduced operators)
    encode    # (params, x)         -> z         full state / series  ->  reduced coords
    decode    # (params, z)         -> x         reduced coords       ->  full state / series
    predict   # (params, z, u, δt)  -> z_next    one step of size δt of the reduced system
end
