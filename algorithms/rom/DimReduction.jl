# Common interface for dimensionality reductions. A `DimReduction` bundles a set of
# parameters with two maps that both take those parameters explicitly as their first
# argument, so the same interface serves both the linear POD reduction (where
# `params` holds a projection basis) and the learned autoencoder (where `params`
# holds encoder/decoder weights):
#   encode(params, x)   full state or time series  ->  reduced coordinates,
#   decode(params, z)   reduced coordinates        ->  full state or time series.
# It is the `ReducedModel` interface **without** `predict`: a reduction only maps
# between the full and reduced spaces and never advances the state in time. Both
# `encode`/`decode` act column-wise, mapping a single state (length-n vector) or a
# whole time series (n×T matrix of column states) in one call. Each reduction in this
# directory constructs its own `DimReduction`.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

struct DimReduction
    params    # parameters of the maps (e.g. a projection basis or network weights)
    encode    # (params, x)  -> z    full state / series  ->  reduced coordinates
    decode    # (params, z)  -> x    reduced coordinates  ->  full state / series
end
