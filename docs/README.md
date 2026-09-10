# Documentation

Detailed write-ups of the algorithms in this collection: the governing equations, the discretisation or model interface, the shared building blocks, and the accompanying scripts.

## Documents

| Topic | File | Description |
| 
| **Ordinary Differential Equations** | [ode.md](ode.md) | Lorenz, Duffing, and Rössler systems; the shared `ODEModel` interface and linearly-implicit trapezoidal time stepping |
| **Partial Differential Equations** | [pde.md](pde.md) | PCB and SLM heat conduction; the shared `PDEModel` interface, finite element / finite volume discretisation, and boundary conditions |
| **Reduced-Order Modelling** | [rom.md](rom.md) | Dimension reduction (POD, autoencoders) and dimension + model reduction (Krylov, POD + Galerkin, operator inference) for the PCB heat problem |
| **Machine Learning** | [ml.md](ml.md) | Learning a vector field from data with a neural network or polynomial model, and the three training schemes (finite difference, Sobolev, neural-ODE/autoregressive) |
| **Uncertainty Quantification** | [uq.md](uq.md) | A Gaussian-process accuracy map predicting a trained model's own rollout error from the current state |

## Student Exercises

Exercises building on the algorithms above live in [exercises/](../exercises/README.md), with one task list per topic: [ml_exercises.md](../exercises/ml_exercises.md), [rom_exercises.md](../exercises/rom_exercises.md), and [uq_exercises.md](../exercises/uq_exercises.md).
