# Physics-Informed Benchmarks

This document describes the benchmark harnesses that validate the physics-informed components
introduced in PR #449 and Issue #400. The benchmarks focus on reproducible PDE baselines and
operator-learning evaluations using simple, deterministic data generation.

## PDE Benchmarks

The PDE benchmark suite provides reference baselines using explicit finite-difference solvers
for 1D Burgers' and Allen-Cahn equations. These baselines are designed for correctness checks
and regression testing rather than large-scale accuracy studies.

### Burgers' Equation

```csharp
using AiDotNet.PhysicsInformed.Benchmarks;

var options = new BurgersBenchmarkOptions
{
    SpatialPoints = 64,
    TimeSteps = 200,
    FinalTime = 1.0,
    Viscosity = 0.01
};

PdeBenchmarkResult result = PdeBenchmarkSuite.RunBurgers(
    options,
    (x, t) => /* model prediction u(x, t) */ 0.0);
```

### Allen-Cahn Equation

```csharp
using AiDotNet.PhysicsInformed.Benchmarks;

var options = new AllenCahnBenchmarkOptions
{
    SpatialPoints = 64,
    TimeSteps = 200,
    FinalTime = 1.0,
    Epsilon = 0.01
};

PdeBenchmarkResult result = PdeBenchmarkSuite.RunAllenCahn(
    options,
    (x, t) => /* model prediction u(x, t) */ 0.0);
```

Both benchmarks return L2 and max error against the finite-difference baseline at the final
time step. Use these results to compare PINN predictions with a traditional solver.

## Operator Learning Benchmark

Operator learning benchmarks use a synthetic smoothing operator applied to random Fourier
series inputs. The goal is to ensure operator learners (e.g., FNO, DeepONet) can reproduce
the moving-average target.

```csharp
using AiDotNet.PhysicsInformed.Benchmarks;

var options = new OperatorBenchmarkOptions
{
    SpatialPoints = 64,
    SampleCount = 32,
    MaxFrequency = 3,
    SmoothingWindow = 5,
    Seed = 42
};

OperatorBenchmarkResult result = OperatorBenchmarkSuite.RunSmoothingOperatorBenchmark(
    options,
    input => /* model prediction */ input);
```

## Real Operator Benchmarks

Two standard operator-learning benchmarks are included with deterministic dataset generation.
Both use simple finite-difference baselines for correctness and regression tracking.

### Poisson Operator (2D)

```csharp
using AiDotNet.PhysicsInformed.Benchmarks;

var options = new PoissonOperatorBenchmarkOptions
{
    GridSize = 32,
    SampleCount = 8,
    MaxFrequency = 3,
    MaxIterations = 2000,
    Tolerance = 1e-6,
    Seed = 42
};

OperatorBenchmarkResult result = OperatorBenchmarkSuite.RunPoissonOperatorBenchmark(
    options,
    forcing => /* model prediction */ forcing);
```

To access the deterministic dataset directly:

```csharp
OperatorDataset2D dataset = OperatorBenchmarkSuite.GeneratePoissonDataset(options);
```

### Darcy Flow Operator (2D)

```csharp
using AiDotNet.PhysicsInformed.Benchmarks;

var options = new DarcyOperatorBenchmarkOptions
{
    GridSize = 32,
    SampleCount = 8,
    MaxFrequency = 3,
    MaxIterations = 3000,
    Tolerance = 1e-6,
    ForcingValue = 1.0,
    LogPermeabilityScale = 0.5,
    Seed = 42
};

OperatorBenchmarkResult result = OperatorBenchmarkSuite.RunDarcyOperatorBenchmark(
    options,
    permeability => /* model prediction */ permeability);
```

To access the deterministic dataset directly:

```csharp
OperatorDataset2D dataset = OperatorBenchmarkSuite.GenerateDarcyDataset(options);
```

## Baselines and Metrics

- PDE baselines: explicit finite-difference time stepping with periodic boundaries.
- Operator baselines: moving-average smoothing plus finite-difference solvers for Poisson and Darcy.
- Metrics: MSE, L2 error, relative L2 error, and max absolute error.

Use the provided harnesses to track regressions and verify that physics-informed models
retain expected numerical behavior across releases.
