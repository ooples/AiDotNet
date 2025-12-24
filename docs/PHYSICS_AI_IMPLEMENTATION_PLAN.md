# Physics-Informed AI Module - Comprehensive Implementation Plan

## Executive Summary

**Goal**: Make physics neural networks work exactly like regular neural networks in AiDotNet.

**What We Fixed**: The physics networks (Hamiltonian, Lagrangian) had custom, complicated training code that didn't match how other networks work. We simplified them to follow the same patterns as `FeedForwardNeuralNetwork`.

**Why This Matters**: When all networks follow the same pattern:
- Easier to maintain (fix one bug, fix it everywhere)
- Easier to understand (learn one pattern, understand all networks)
- Fewer bugs (standard code is better tested)
- Better performance (no unnecessary fallback calculations)

---

## Understanding Neural Network Training (For Beginners)

### What Happens When You Train a Neural Network?

Think of training a neural network like teaching a child to throw a ball at a target:

1. **Forward Pass** - The child throws the ball (network makes a prediction)
2. **Loss Calculation** - We measure how far the ball landed from the target (calculate error)
3. **Backward Pass** - We figure out what went wrong (compute gradients)
4. **Parameter Update** - The child adjusts their technique (update weights)

This happens over and over until the child gets good at hitting the target.

### The Standard Training Pattern

Every neural network in AiDotNet follows this exact pattern:

```csharp
public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
{
    // Step 1: Make a prediction
    var prediction = Forward(input);

    // Step 2: Calculate how wrong we were
    var loss = _lossFunction.CalculateLoss(prediction, expectedOutput);

    // Step 3: Figure out how to improve (backpropagation)
    var gradient = _lossFunction.CalculateDerivative(prediction, expectedOutput);
    Backward(gradient);

    // Step 4: Update the network's parameters
    _optimizer.UpdateParameters(Layers);
}
```

**Key Point**: The "Backward" pass computes gradients by going through each layer in reverse order. Each layer knows how to compute its own gradients. This is called **backpropagation**.

### What Was Wrong With Physics Networks?

The old physics networks had a "safety net" that used **finite difference gradients** when backpropagation failed:

```csharp
// OLD, WRONG WAY
if (TryBackpropagate(gradient, out gradients))
{
    ApplyGradients(gradients);
}
else
{
    // This is like saying "if throwing fails, calculate physics manually"
    gradients = ComputeFiniteDifferenceGradients(...);  // SLOW and unnecessary!
    ApplyGradients(gradients);
}
```

**Why this was bad:**
1. **Slow**: Finite difference requires many forward passes to approximate gradients
2. **Inaccurate**: It's an approximation, not exact like backpropagation
3. **Confusing**: Different from how other networks work
4. **Unnecessary**: If a layer can't backpropagate, that's a bug to fix, not work around

---

## Phase 1: Standard Neural Network Patterns (COMPLETED)

### 1.1 The Golden Pattern: FeedForwardNeuralNetwork

All neural networks must follow this structure:

```csharp
public class MyNeuralNetwork<T> : NeuralNetworkBase<T>
{
    // 1. USE LAYERHELPER FOR DEFAULT LAYERS
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // User provided custom layers
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Use standard layer creation from LayerHelper
            Layers.AddRange(LayerHelper<T>.CreateDefaultMyNetworkLayers(Architecture));
        }
    }

    // 2. STANDARD FORWARD PASS
    public Tensor<T> Forward(Tensor<T> input)
    {
        Tensor<T> output = input;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }
        return output;
    }

    // 3. STANDARD BACKWARD PASS
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        Tensor<T> gradient = outputGradient;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }
        return gradient;
    }

    // 4. STANDARD TRAIN METHOD
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        IsTrainingMode = true;

        var prediction = Forward(input);
        LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

        var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        Backward(Tensor<T>.FromVector(outputGradient));
        _optimizer.UpdateParameters(Layers);

        IsTrainingMode = false;
    }
}
```

### 1.2 Using LayerHelper for Default Layers

**Why LayerHelper?**

LayerHelper is like a factory that creates the right layers for each type of network. This keeps layer creation logic in one place instead of scattered across many network classes.

**Location**: `src/Helpers/LayerHelper.cs`

**Pattern**:
```csharp
// In LayerHelper.cs
public static IEnumerable<ILayer<T>> CreateDefaultHamiltonianLayers(
    NeuralNetworkArchitecture<T> architecture,
    int hiddenLayerCount = 3,
    int hiddenLayerSize = 64)
{
    // Validation
    if (architecture.OutputSize != 1)
        throw new ArgumentException("Hamiltonian networks require scalar output.");

    // Create layers with appropriate activations
    var hiddenActivation = new TanhActivation<T>() as IActivationFunction<T>;

    yield return new DenseLayer<T>(inputSize, hiddenLayerSize, hiddenActivation);
    // ... more layers ...
    yield return new DenseLayer<T>(hiddenLayerSize, 1, new IdentityActivation<T>());
}
```

**Existing LayerHelper Methods**:
- `CreateDefaultFeedForwardLayers()` - Standard feed-forward networks
- `CreateDefaultDeepQNetworkLayers()` - Reinforcement learning
- `CreateDefaultNodeClassificationLayers()` - Graph neural networks
- `CreateDefaultHamiltonianLayers()` - Hamiltonian physics networks (NEW)
- `CreateDefaultLagrangianLayers()` - Lagrangian physics networks (NEW)

### 1.3 IEngine: Already Inherited, Never Injected

**What is IEngine?**

IEngine provides optimized math operations (like matrix multiplication) that can use SIMD or GPU acceleration.

**The Rule**: Never add IEngine as a constructor parameter. It's already available through inheritance:

```csharp
// In NeuralNetworkBase<T>
protected IEngine Engine => AiDotNetEngine.Current;

// In your network class, just use it:
var result = Engine.MatrixMultiply(a, b);
```

**Why?** Engine is a global service. Injecting it through constructors:
- Adds unnecessary complexity
- Creates inconsistency (some networks take it, some don't)
- Doesn't match how other networks work

### 1.4 Refactoring Status

| Network | Status | Changes Made |
|---------|--------|--------------|
| `HamiltonianNeuralNetwork` | COMPLETED | Removed `EnableAutodiffOnLayers()`, removed FD fallbacks, uses `LayerHelper.CreateDefaultHamiltonianLayers()` |
| `LagrangianNeuralNetwork` | COMPLETED | Removed `EnableAutodiffOnLayers()`, removed FD fallbacks, uses `LayerHelper.CreateDefaultLagrangianLayers()` |
| `PhysicsInformedNeuralNetwork` | COMPLETED | Already correct - uses `NeuralNetworkDerivatives` for physics (not training) |

---

## Understanding Physics Gradients vs Training Gradients

This is a crucial distinction that was causing confusion:

### Training Gradients (For Learning)

**Question**: "How should I adjust the network's weights to make better predictions?"

**Computed via**: Standard backpropagation through layers

**Used in**: The `Train()` method

**Symbol**: dLoss/dWeights (derivative of loss with respect to weights)

```csharp
// This is TRAINING - adjusting weights to reduce error
public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
{
    var prediction = Forward(input);
    var loss = _lossFunction.CalculateLoss(prediction, expectedOutput);
    Backward(outputGradient);  // This computes dLoss/dWeights
    _optimizer.UpdateParameters(Layers);  // This applies the update
}
```

### Physics Gradients (For Simulation)

**Question**: "Given the current state of a physical system, what are the forces/velocities?"

**Computed via**: `NeuralNetworkDerivatives` utility (uses finite difference or autodiff)

**Used in**: Physics simulation methods like `Simulate()` or `ComputeTimeDerivative()`

**Symbol**: dH/dq (derivative of Hamiltonian with respect to position)

```csharp
// This is PHYSICS - computing forces from energy gradients
public T[] ComputeTimeDerivative(T[] state)
{
    // We need dH/dq and dH/dp to apply Hamilton's equations
    T[,] gradient = NeuralNetworkDerivatives<T>.ComputeGradient(this, state, 1);

    // Hamilton's equations: dq/dt = dH/dp, dp/dt = -dH/dq
    for (int i = 0; i < n; i++)
    {
        derivative[i] = gradient[0, n + i];       // dq/dt = dH/dp
        derivative[n + i] = -gradient[0, i];      // dp/dt = -dH/dq
    }
    return derivative;
}
```

### The Key Insight

- **Training gradients** are computed with respect to the network's **weights** (to make the network learn)
- **Physics gradients** are computed with respect to the **inputs** (to simulate the physical system)

These are completely different operations with different purposes!

---

## Phase 2: PhysicsInformedLoss Design (Already Correct)

The `PhysicsInformedLoss` class was already designed correctly:

1. **Spatial derivatives** (like dT/dx for temperature) - Uses `NeuralNetworkDerivatives`
2. **PDE residual gradients** - Built-in PDEs implement `IPDEResidualGradient<T>`
3. **FD fallback** - Only for custom PDEs that don't provide analytic gradients

This is the correct design because:
- Spatial derivatives ARE physics gradients (input-space)
- The FD fallback here is reasonable for user-defined PDEs

---

## Phase 3: Future Work - Expanding PDE Library

### PDEs Already Implemented

| PDE | Description | Has Analytic Gradient? |
|-----|-------------|------------------------|
| Heat Equation | Heat diffusion (dT/dt = k*d2T/dx2) | Yes |
| Poisson Equation | Electrostatics (d2u/dx2 = f) | Yes |
| Wave Equation | Sound/light waves (d2u/dt2 = c2*d2u/dx2) | Yes |
| Burgers Equation | Shock waves (du/dt + u*du/dx = v*d2u/dx2) | Yes |
| Allen-Cahn Equation | Phase transitions | Yes |
| Navier-Stokes | 2D incompressible fluid dynamics | Yes |
| Maxwell's Equations | 2D TE mode electromagnetics | Yes |
| Schrodinger Equation | 1D time-dependent quantum mechanics | Yes |
| Black-Scholes | Option pricing in finance | Yes |
| Linear Elasticity | 2D Navier-Cauchy solid mechanics | Yes |
| Advection-Diffusion | 1D/2D transport phenomena | Yes |
| Korteweg-de Vries | Soliton waves (KdV equation) | Yes |

---

## Implementation Checklist

### Sprint 1: Align with Standard Patterns (COMPLETED)
- [x] PhysicsInformedNeuralNetwork uses NeuralNetworkDerivatives for physics gradients
- [x] IEngine available via inheritance (no constructor injection)
- [x] PhysicsInformedLoss has correct design
- [x] Refactor HamiltonianNeuralNetwork to use LayerHelper and standard Train
- [x] Refactor LagrangianNeuralNetwork to use LayerHelper and standard Train
- [x] Remove `EnableAutodiffOnLayers()` calls from Hamiltonian/Lagrangian
- [x] Remove FD fallbacks from Hamiltonian/Lagrangian Train methods
- [x] Add `CreateDefaultHamiltonianLayers()` to LayerHelper
- [x] Add `CreateDefaultLagrangianLayers()` to LayerHelper
- [x] Add deprecation documentation to FiniteDifferenceGradient.cs

### Sprint 1.5: Remaining Network Refactoring (COMPLETED)
All physics networks now use the standard pattern:
- [x] Refactor `UniversalDifferentialEquations.cs` to use standard Train pattern
- [x] Refactor `DeepOperatorNetwork.cs` to use standard Train pattern
- [x] Refactor `FourierNeuralOperator.cs` to use standard Train pattern
- [x] Refactor `VariationalPINN.cs` to use standard Train pattern
- [x] Refactor `DeepRitzMethod.cs` to use standard Train pattern
- [x] Refactor `PhysicsInformedNeuralNetwork.cs` to use standard Train pattern
- [x] Add `CreateDefaultUniversalDELayers()` to LayerHelper
- [x] Add `CreateDefaultDeepOperatorNetworkLayers()` to LayerHelper
- [x] Add `CreateDefaultFourierNeuralOperatorLayers()` to LayerHelper
- [x] Add `CreateDefaultVariationalPINNLayers()` to LayerHelper
- [x] Add `CreateDefaultDeepRitzLayers()` to LayerHelper
- [x] Add `CreateDefaultPINNLayers()` to LayerHelper

### Sprint 2: Core PDEs (COMPLETED)
- [x] Implement Navier-Stokes equation
- [x] Implement Maxwell's equations
- [x] Implement Schrodinger equation
- [x] Add PDE unit tests with known analytical solutions

### Sprint 3: Advanced Features (COMPLETED)
- [x] Multi-scale physics support (IMultiScalePDE interface, MultiScalePINN implementation)
- [x] Inverse problems (IInverseProblem interface, InverseProblemPINN for parameter identification)
- [x] Unit tests for multi-scale and inverse problem features

**Multi-scale Physics:**
- `IMultiScalePDE<T>` interface for PDEs with multiple length/time scales
- `MultiScalePINN<T>` implementation with support for:
  - Sequential (coarse-to-fine) and simultaneous training
  - Adaptive scale weighting
  - Scale coupling between different resolution levels

**Inverse Problems:**
- `IInverseProblem<T>` interface for parameter identification
- `InverseProblemPINN<T>` implementation with support for:
  - Multiple regularization types (L2 Tikhonov, L1 Lasso, Elastic Net, Bayesian, Total Variation)
  - Parameter bounds and validation
  - Uncertainty estimation for identified parameters
  - Separate learning rates for network and parameters

### Sprint 4: Multi-Fidelity and Domain Decomposition (COMPLETED)
- [x] Multi-fidelity physics-informed learning (`MultiFidelityPINN<T>`)
- [x] Domain decomposition for large-scale problems (`DomainDecompositionPINN<T>`)
- [x] Extended training history interfaces (`IMultiFidelityTrainingHistory<T>`, `IDomainDecompositionTrainingHistory<T>`)
- [x] Unit tests for Sprint 4 features

**Multi-Fidelity PINN:**
- `MultiFidelityPINN<T>` implementation with support for:
  - Combining low-fidelity (cheap) and high-fidelity (expensive) data
  - Nonlinear correlation learning between fidelity levels
  - Pretraining on low-fidelity data before joint optimization
  - Optional freezing of low-fidelity network after pretraining
  - Customizable fidelity weights

**Domain Decomposition PINN:**
- `DomainDecompositionPINN<T>` implementation with support for:
  - Non-overlapping domain decomposition
  - Automatic interface detection between subdomains
  - Schwarz-style iterative training
  - Interface continuity enforcement
  - Per-subdomain network customization

### Sprint 5: GPU Acceleration (COMPLETED)
- [x] GPU acceleration for physics-informed training
- [x] `IGpuAcceleratedPINN<T>` interface for GPU-capable networks
- [x] `GpuPINNTrainingOptions` configuration class with presets (Default, HighEnd, LowMemory, CpuOnly)
- [x] `GpuPINNTrainer<T>` for GPU-accelerated PINN training
- [x] `GpuTrainingHistory<T>` with GPU-specific metrics
- [x] Unit tests for GPU acceleration features (22 tests)

**GPU Acceleration Features:**
- `IGpuAcceleratedPINN<T>` interface for marking GPU-capable PINNs
- `GpuPINNTrainingOptions` with configuration for:
  - Batch size optimization for GPU (default 1024)
  - Parallel derivative computation
  - Asynchronous GPU transfers
  - Mixed precision support (FP16)
  - Pinned memory for faster transfers
  - Multi-stream execution
- `GpuPINNTrainer<T>` providing:
  - Automatic GPU detection and fallback
  - GPU memory usage tracking
  - Training time metrics
  - Easy integration with existing PINNs

### Sprint 6: Advanced PDE Library (COMPLETED)
- [x] Black-Scholes equation for option pricing in finance
- [x] Linear Elasticity equation (2D Navier-Cauchy) for solid mechanics
- [x] Advection-Diffusion equation (1D and 2D) for transport phenomena
- [x] Korteweg-de Vries equation for soliton/solitary wave modeling
- [x] Third-order derivative support in `PDEDerivatives<T>` and `PDEResidualGradient<T>`
- [x] Unit tests for all new PDEs (32 tests)

**New PDE Features:**
- `BlackScholesEquation<T>`: Option pricing with volatility and risk-free rate parameters
- `LinearElasticityEquation<T>`: 2D solid mechanics with Lamé parameters or engineering constants (E, ν)
- `AdvectionDiffusionEquation<T>`: Transport phenomena with both advection and diffusion
  - Supports 1D and 2D configurations
  - Configurable velocity field, diffusion coefficient, and source term
- `KortewegDeVriesEquation<T>`: Nonlinear wave propagation with soliton solutions
  - Factory methods for canonical (α=6, β=1) and physical (α=1, β=1) forms
  - Third-order spatial derivative support

---

## Success Criteria

1. **Pattern Alignment**: All physics networks follow the `FeedForwardNeuralNetwork` pattern
2. **LayerHelper Integration**: All networks use `LayerHelper.CreateDefault*Layers()`
3. **No Forced Autodiff**: Removed all `EnableAutodiffOnLayers()` calls
4. **No Training FD**: Removed finite difference fallbacks from `Train()` methods
5. **Clean Separation**: Physics gradients (simulation) vs Training gradients (learning)

---

## Appendix: Code Comparison

### Before (WRONG)

```csharp
public class HamiltonianNeuralNetwork<T> : NeuralNetworkBase<T>
{
    // Custom layer creation (should use LayerHelper)
    private void AddDefaultLayers(int inputSize, int outputSize, int[] hiddenLayerSizes)
    {
        // ... custom logic scattered in network class ...
    }

    // Forced autodiff (not needed)
    private void EnableAutodiffOnLayers()
    {
        foreach (var layer in Layers)
            if (layer is LayerBase<T> layerBase)
                layerBase.UseAutodiff = true;
    }

    // Training with FD fallback (overcomplicated)
    public override void Train(...)
    {
        if (TryBackpropagate(gradient, out gradients))
            ApplyGradients(gradients);
        else
            gradients = ComputeFiniteDifferenceGradients(...);  // BAD
            ApplyGradients(gradients);
    }
}
```

### After (CORRECT)

```csharp
public class HamiltonianNeuralNetwork<T> : NeuralNetworkBase<T>
{
    // Use LayerHelper (layer logic centralized)
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultHamiltonianLayers(Architecture));
        }
    }

    // Standard train (same as all other networks)
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        IsTrainingMode = true;
        var prediction = Forward(input);
        LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        Backward(Tensor<T>.FromVector(outputGradient));
        _optimizer.UpdateParameters(Layers);
        IsTrainingMode = false;
    }

    // Physics simulation uses NeuralNetworkDerivatives (correct!)
    public T[] ComputeTimeDerivative(T[] state)
    {
        T[,] gradient = NeuralNetworkDerivatives<T>.ComputeGradient(this, state, 1);
        // Apply Hamilton's equations...
    }
}
```
