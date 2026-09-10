# Machine-Learning Exercises


## Contents

- [Exercise 1 — Inferring dynamics for other provided ODEs](#exercise-1--inferring-dynamics-for-other-provided-odes)
- [Exercise 2 — Neural architecture sweep](#exercise-2--neural-architecture-sweep)
- [Exercise 3 — GPU vs. CPU training](#exercise-3--gpu-vs-cpu-training)
- [Exercise 4 — Robustness to measurement noise](#exercise-4--robustness-to-measurement-noise)
- [Exercise 5 — Higher-order derivative estimates and noise robustness](#exercise-5--higher-order-derivative-estimates-and-noise-robustness)
- [Exercise 6 — Optimizer choice: finite-difference training](#exercise-6--optimizer-choice-finite-difference-training)
- [Exercise 7 — Optimizer choice: rollout training](#exercise-7--optimizer-choice-rollout-training)
- [Exercise 8 — Two-stage training: finite-difference then rollout refinement](#exercise-8--two-stage-training-finite-difference-then-rollout-refinement)

## Exercises

### Exercise 1 — Inferring dynamics for other provided ODEs

[Lorenz_NN_finitediff.jl](../scripts/ml/Lorenz_NN_finitediff.jl) fits a neural
vector field to the Lorenz system. Repeat the same recipe — first-order forward finite-
difference labels, the same network architecture, the same train/validation/test split
and rollout evaluation, for the other ODEs provided in [models/ode/](../models/ode):
the **Rössler system** and the **Duffing oscillator**. Neither has a pre-existing
training-trajectory file like Lorenz does, so generate a small ensemble inline. How does
rollout accuracy compare across the three systems?


### Exercise 2 — Neural architecture sweep

Using [Lorenz_NN_finitediff.jl](../scripts/ml/Lorenz_NN_finitediff.jl) as a
starting point, sweep the network depth (2, 3, 4 layers) and width (20, 40, 60 neurons)
over all 9 combinations, keeping everything else fixed. Which architecture gives the
best long-horizon (rollout) accuracy? Is bigger always better?


### Exercise 3 — GPU vs. CPU training

Translate [Lorenz_NN_finitediff.jl](../scripts/ml/Lorenz_NN_finitediff.jl)'s
training loop to run on the GPU (`gpu_device()`, moving parameters, batch, control, and
standardisation constants together) and compare wall-clock training time against the
identical CPU run. Does the learned model differ in accuracy?


### Exercise 4 — Robustness to measurement noise

Add uniform, coordinate-wise additive noise to the training trajectories,
bounded by ±1%, ±2%, and ±5% of the *global* (whole-dataset) absolute maximum of each
coordinate,  before forming the training data, and measure how the rollout loss
degrades for both the neural network and the polynomial model, each trained with the
finite-difference objective. Which model degrades more gracefully?


### Exercise 5 — Higher-order derivative estimates and noise robustness

Lorenz_NN_finitediff.jl uses the first-order forward difference
$d_n = (x_{n+1} - x_n)/\delta t$ as its derivative label. Try three higher-order
variants instead - e.g., a second-order central difference,
$d_n = (x_{n+1} - x_{n-1})/(2\delta t)$, a fourth-order central difference,
$d_n = (-x_{n+2} + 8 x_{n+1} - 8 x_{n-1} + x_{n-2})/(12\delta t)$, and a sixth-order
central difference,
$d_n = (x_{n+3} - 9 x_{n+2} + 45 x_{n+1} - 45 x_{n-1} + 9 x_{n-2} - x_{n-3})/(60\delta t)$ -
and repeat the noise-robustness sweep of Exercise 4 for each. Does a higher-order
(lower truncation error) derivative label make the network more or less robust to
noise?


### Exercise 6 — Optimizer choice: finite-difference training

Repeat the noise-robustness sweep of Exercise 4, but train each noise level
twice, once with `Adam` and once with `LBFGS`, for both the neural network and the
polynomial model. Does the optimizer choice change the conclusions of Exercise 4? How
do the optimizers compare in wall-clock cost?


### Exercise 7 — Optimizer choice: rollout training

Repeat Exercise 6, but this time train with the rollout (autoregressive /
neural-ODE) objective instead of finite differences, for both model classes. Does
training on rollouts rather than finite differences change the noise sensitivity seen
in Exercises 4 and 6?


### Exercise 8 — Two-stage training: finite-difference then rollout refinement

Because the weights are decoupled from the training strategy
(`θ0` is an explicit argument to both `train_model_residual` and
`train_model_autoregressive`), a model trained one way can be *handed to* the other
training function to continue training. Fit each noisy case first with the cheap
finite-difference regression, then continue training the *same weights* with a
(much smaller, 25%-sized) rollout refinement stage. Does refinement help, and does it
help both model classes equally?

