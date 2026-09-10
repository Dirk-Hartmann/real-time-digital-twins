# Data

This folder holds generated training datasets only; nothing under it (besides this file) is tracked by git (see `.gitignore`) to ensure a lean repository. Every file here is produced by a script and can be regenerated at any time by rerunning that script.

- `Lorenz_training.bson` is created by
  [scripts/ode/Lorenz_training_data.jl](../scripts/ode/Lorenz_training_data.jl): an ensemble of 100 Lorenz trajectories from random initial conditions near the attractor, stored as the time vector `t` and the array `X` (state × time × trajectory). 
- `PCB_training.bson` is created by
  [scripts/pde/PCB_training_data.jl](../scripts/pde/PCB_training_data.jl): an ensemble of 10 transient PCB thermal fields, each driven by a fresh pair of random cooling / heating signals, stored as the time vector `t`, the temperature array `T` (node × time × simulation), the control array `u` (input × time × simulation), and the grid metadata `nelx`, `nely`, `δx`.
- `SLM_training.bson` is created by
  [scripts/pde/SLM_transient.jl](../scripts/pde/SLM_transient.jl): a single transient SLM thermal field driven by a moving Gaussian laser spot with fixed initial position and velocity, stored as the time vector `t`, the temperature array `T` (node × time × 1), the laser position array `u` (input × time × 1, `u[1,:]`/`u[2,:]` the x/y spot centre), and the grid metadata `nelx`, `nely`, `δx`.

