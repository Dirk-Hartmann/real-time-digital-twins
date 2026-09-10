# Splits an ensemble of `ntraj` trajectories into disjoint training / validation / test
# index ranges from user-chosen percentages: `val_frac` and `test_frac` set aside the
# validation and test trajectories (each rounded to the nearest trajectory count), and
# training receives the remainder, so the three ranges always partition `1:ntraj`.
#
# Copyright (c) 2026 Dirk Hartmann, TU Darmstadt

# --- Partition 1:ntraj into train / val / test ranges from percentages ---
function split_trajectories(ntraj; val_frac, test_frac)
    nval  = round(Int, val_frac * ntraj)
    ntest = round(Int, test_frac * ntraj)
    ntrain = ntraj - ntest - nval
    println("Total trajectories: $ntraj | Train: $ntrain | Validation: $nval | Test: $ntest\n")
    return 1:ntrain, ntrain .+ (1:nval), (ntrain + nval) .+ (1:ntest)
end
