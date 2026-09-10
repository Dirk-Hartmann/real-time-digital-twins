# Uncertainty-Quantification Exercises


## Contents

- [Exercise 1 — A Gaussian-process error estimator for the operator-inference ROM](#exercise-1--a-gaussian-process-error-estimator-for-the-operator-inference-rom)

## Exercises

### Exercise 1 — A Gaussian-process error estimator for the operator-inference ROM

[uq.md](uq.md) fits a Gaussian process (GP) that predicts, from the current
state alone, how large a trained model's own short-horizon rollout error is about to
be, for the autonomous Lorenz neural predictor. Apply the exact same two algorithms
([RolloutError.jl](../algorithms/uq/RolloutError.jl),
[GPErrorModel.jl](../algorithms/uq/GPErrorModel.jl)) to the **driven** operator-inference
reduced model of [PCB_OperatorInference.jl](../scripts/rom/PCB_OperatorInference.jl):
fit the reduced operators $(A, C)$ from one reference simulation as that script does,
then fit a GP to $(\zeta_0, \mathrm{mse})$ pairs from 5-step ($0.5\,\mathrm{s}$) reduced
rollout chunks scored on the *other* simulations of the PCB ensemble (held out from the
fit, so the pairs reflect genuine out-of-sample accuracy) under each one's own recorded
drive $u=[c;h]$. Choose one further simulation purely for testing and plot its
reduced-coordinate trajectories (the true one, after POD projection, against the model
fully simulated under the recorded drive), the true vs GP-estimated short-horizon
reduced rollout error, and the resulting full-field relative error over the complete
horizon.
