# Optimizer DataLoader Integration Checklist

This document tracks the progress of integrating the DataLoader API into all gradient-based optimizers.

## Integration Pattern

Each optimizer requires:
1. Add `BatchSize` property to options class (default: 32)
2. Change iteration loop to epoch-based loop
3. Add `NotifyEpochStart(epoch)` at start of each epoch
4. Create batcher: `var batcher = CreateBatcher(inputData, _options.BatchSize)`
5. Inner loop: `foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())`
6. Move evaluation to after batch loop (end of epoch)
7. Add DataLoader Integration documentation in remarks

---

## First-Order Gradient Optimizers

### Core Optimizers
- [x] **StochasticGradientDescentOptimizer** - Refactored (BatchSize default: 1)
- [x] **MiniBatchGradientDescentOptimizer** - Already uses DataLoader
- [x] **GradientDescentOptimizer** - Refactored (BatchSize default: 32)

### Adam Family
- [x] **AdamOptimizer** - Refactored
- [x] **AdamWOptimizer** - Refactored
- [x] **NadamOptimizer** - Refactored
- [x] **AMSGradOptimizer** - Refactored
- [x] **AdaMaxOptimizer** - Refactored

### Momentum Family
- [x] **MomentumOptimizer** - Refactored
- [x] **NesterovAcceleratedGradientOptimizer** - Refactored

### Adaptive Learning Rate Family
- [x] **AdagradOptimizer** - Refactored
- [x] **AdaDeltaOptimizer** - Refactored
- [x] **RootMeanSquarePropagationOptimizer** (RMSprop) - Refactored

### Modern/Specialized Optimizers
- [x] **LionOptimizer** - Refactored
- [x] **FTRLOptimizer** - Refactored
- [x] **ProximalGradientDescentOptimizer** - Refactored
- [x] **ADMMOptimizer** - Refactored (epoch notification only, full batch)
- [x] **CoordinateDescentOptimizer** - Refactored (epoch notification only, full batch)

---

## Second-Order Optimizers (Full-Batch Only)

These optimizers require full-batch gradients due to Hessian/quasi-Newton requirements.
They use `BatchSize = -1` (full batch) and have `NotifyEpochStart(epoch)` for curriculum learning compatibility.

- [x] **ConjugateGradientOptimizer** - Refactored (epoch notification, full batch)
- [x] **BFGSOptimizer** - Refactored (epoch notification, full batch for Hessian approximation)
- [x] **LBFGSOptimizer** - Refactored (epoch notification, full batch for limited-memory BFGS)
- [x] **DFPOptimizer** - Refactored (epoch notification, full batch for inverse Hessian approximation)
- [x] **NewtonMethodOptimizer** - Refactored (epoch notification, full batch for Hessian computation)
- [x] **LevenbergMarquardtOptimizer** - Refactored (epoch notification, full batch for Jacobian)
- [x] **TrustRegionOptimizer** - Refactored (epoch notification, full batch for quadratic model)

---

## Non-Gradient-Based Optimizers (Not Applicable)

These optimizers do not use gradients and don't need DataLoader integration:

- N/A **GeneticAlgorithmOptimizer** - Evolutionary algorithm
- N/A **ParticleSwarmOptimizer** - Swarm intelligence
- N/A **SimulatedAnnealingOptimizer** - Probabilistic metaheuristic
- N/A **DifferentialEvolutionOptimizer** - Evolutionary algorithm
- N/A **AntColonyOptimizer** - Swarm intelligence
- N/A **CMAESOptimizer** - Evolutionary strategy
- N/A **TabuSearchOptimizer** - Metaheuristic
- N/A **BayesianOptimizer** - Bayesian optimization
- N/A **NelderMeadOptimizer** - Simplex method
- N/A **PowellOptimizer** - Direction set method
- N/A **NormalOptimizer** - Statistical optimization

---

## Progress Summary

| Category | Completed | Total | Percentage |
|----------|-----------|-------|------------|
| Core Optimizers | 3 | 3 | 100% |
| Adam Family | 5 | 5 | 100% |
| Momentum Family | 2 | 2 | 100% |
| Adaptive LR Family | 3 | 3 | 100% |
| Modern/Specialized | 5 | 5 | 100% |
| Second-Order | 7 | 7 | 100% |
| **Total All** | **25** | **25** | **100%** |

---

## Batching Semantics Summary

| Optimizer Type | Default BatchSize | Batching Support |
|----------------|-------------------|------------------|
| First-Order (SGD, Adam, etc.) | 32 | Full mini-batch support |
| Stochastic GD | 1 | Single-sample batching |
| ADMM, Coordinate Descent | -1 (full) | Epoch notification only |
| Second-Order (BFGS, Newton, etc.) | -1 (full) | Epoch notification only |

**Note**: Second-order optimizers use full-batch gradients because they build Hessian or
quasi-Hessian approximations that require consistent gradient information between iterations.
Mini-batching would introduce noise that disrupts these approximations.

---

## Unit Tests (Phase 6)

Comprehensive unit tests created for the DataLoader infrastructure:

| Test File | Tests | Description |
|-----------|-------|-------------|
| `OptimizationDataBatcherTests.cs` | 29 | Tests for batch creation, shuffling, drop last, seeds |
| `DataSamplerTests.cs` | 45 | Tests for RandomSampler, SequentialSampler, SubsetSampler, CurriculumSampler, WeightedSampler |
| `OptimizerBatchingTests.cs` | 26 | Tests for optimizer options, batch sizes, sampler integration |
| **Total** | **100** | All tests passing |

---

## Performance Benchmarks (Phase 7)

Comprehensive BenchmarkDotNet benchmarks for DataLoader infrastructure:

| Benchmark File | Benchmarks | Description |
|----------------|------------|-------------|
| `DataLoaderBenchmarks.cs` | 28 | Tests for batcher creation, iteration, batch sizes, shuffle overhead, samplers |

**Benchmark Categories:**
- **Batcher Creation**: Measure overhead of creating batchers with/without shuffle
- **Batch Iteration**: Compare iteration performance across dataset sizes (100, 1000, 10000 samples)
- **Batch Size Comparison**: Compare batch sizes 1, 16, 32, 64, 128
- **Shuffle vs No-Shuffle**: Measure shuffle overhead impact
- **Sampler Performance**: RandomSampler, SequentialSampler, CurriculumSampler
- **GetBatchIndices vs GetBatches**: Compare lightweight index-only vs full data extraction
- **DropLast Comparison**: Measure impact of dropping incomplete last batch
- **WeightedSampler**: Balanced weights creation and weighted sampling

**Running Benchmarks:**
```bash
dotnet run -c Release --project AiDotNetBenchmarkTests/AiDotNetBenchmarkTests.csproj -- --filter "*DataLoader*"
```

---

## Last Updated
- Date: 2025-12-24
- Session: Issue #443 DataLoader Integration
- Phase: Phase 7 complete (performance benchmarks added)
