# Issue #400: Physics-Informed Neural Networks (PINNs) - Junior Developer Implementation Guide

## Overview

Physics-Informed Neural Networks (PINNs) combine deep learning with physical laws by encoding differential equations directly into the loss function. Instead of relying purely on data, PINNs constrain predictions to satisfy known physics, making them powerful for scientific computing tasks with limited data.

**Learning Value**: Understanding how to integrate domain knowledge (physics equations) with neural networks, automatic differentiation for computing derivatives, and multi-objective loss functions.

**Estimated Complexity**: Advanced (15-25 hours)

**Prerequisites**:
- Strong understanding of neural networks and backpropagation
- Calculus and differential equations (PDEs/ODEs)
- Automatic differentiation concepts
- Multi-task learning and loss weighting

---

## Educational Objectives

By implementing PINNs, you will learn:

1. **Automatic Differentiation**: How to compute derivatives of network outputs with respect to inputs
2. **Physics Loss Functions**: Encoding differential equations as soft constraints
3. **Residual Minimization**: Training networks to minimize equation residuals
4. **Boundary Conditions**: Enforcing initial/boundary conditions in loss functions
5. **Multi-Objective Optimization**: Balancing data loss, physics loss, and boundary loss
6. **Collocation Points**: Sampling domain points for physics constraint evaluation

---

## Physics Background

### What Are PINNs?

Traditional neural networks learn purely from data:
```
Loss = MSE(predictions, labels)
```

PINNs add physics constraints:
```
Total Loss = Data Loss + Physics Loss + Boundary Loss
```

### Example: Heat Equation

The 1D heat equation describes temperature diffusion:
```
∂u/∂t = α ∂²u/∂x²
```

Where:
- `u(x,t)` is temperature at position x and time t
- `α` is thermal diffusivity

A PINN learns `u(x,t)` by:
1. **Data loss**: Match observed temperatures
2. **Physics loss**: Satisfy `∂u/∂t - α ∂²u/∂x² = 0`
3. **Boundary loss**: Enforce initial/boundary conditions

---

## Architecture Design

### Core Interfaces

```csharp
namespace AiDotNet.PhysicsInformed
{
    /// <summary>
    /// Represents a physics-informed neural network that enforces physical laws
    /// through the loss function.
    /// </summary>
    /// <typeparam name="T">Data type (float, double)</typeparam>
    public interface IPhysicsInformedNetwork<T> where T : struct
    {
        /// <summary>
        /// Forward pass: Computes network output u(x,t,...)
        /// </summary>
        /// <param name="inputs">Spatial/temporal coordinates [batch, inputDim]</param>
        /// <returns>Network predictions u [batch, outputDim]</returns>
        Matrix<T> Forward(Matrix<T> inputs);

        /// <summary>
        /// Computes derivatives of outputs with respect to inputs.
        /// Essential for evaluating differential equations.
        /// </summary>
        /// <param name="inputs">Input coordinates</param>
        /// <param name="order">Derivative order (1 for ∂u/∂x, 2 for ∂²u/∂x²)</param>
        /// <param name="inputIndex">Which input variable to differentiate</param>
        /// <returns>Derivatives [batch, outputDim]</returns>
        Matrix<T> ComputeDerivatives(Matrix<T> inputs, int order, int inputIndex);

        /// <summary>
        /// Evaluates the physics residual (how much the equation is violated).
        /// For heat equation: residual = ∂u/∂t - α ∂²u/∂x²
        /// </summary>
        /// <param name="collocationPoints">Points where physics is enforced</param>
        /// <returns>Residual values (should approach zero)</returns>
        Matrix<T> ComputePhysicsResidual(Matrix<T> collocationPoints);
    }

    /// <summary>
    /// Defines a partial differential equation (PDE) or ordinary differential equation (ODE)
    /// that the PINN must satisfy.
    /// </summary>
    /// <typeparam name="T">Data type</typeparam>
    public interface IPhysicsEquation<T> where T : struct
    {
        /// <summary>
        /// Computes the residual of the physics equation.
        /// </summary>
        /// <param name="inputs">Input coordinates (x, t, ...)</param>
        /// <param name="outputs">Network predictions u(x,t,...)</param>
        /// <param name="derivatives">Dictionary of derivatives (e.g., "du/dt", "d²u/dx²")</param>
        /// <returns>Residual (should be zero when equation is satisfied)</returns>
        Matrix<T> ComputeResidual(
            Matrix<T> inputs,
            Matrix<T> outputs,
            Dictionary<string, Matrix<T>> derivatives);

        /// <summary>
        /// Gets the names of required derivatives for this equation.
        /// </summary>
        List<DerivativeSpec> RequiredDerivatives { get; }
    }

    /// <summary>
    /// Specifies a derivative requirement.
    /// </summary>
    public class DerivativeSpec
    {
        public string Name { get; set; }           // e.g., "du/dt"
        public int InputIndex { get; set; }         // Which input variable (0=x, 1=t)
        public int Order { get; set; }              // 1 for first derivative, 2 for second
    }

    /// <summary>
    /// Defines boundary and initial conditions.
    /// </summary>
    /// <typeparam name="T">Data type</typeparam>
    public interface IBoundaryCondition<T> where T : struct
    {
        /// <summary>
        /// Evaluates the boundary condition residual.
        /// </summary>
        /// <param name="boundaryPoints">Points on the boundary</param>
        /// <param name="predictions">Network predictions at boundary</param>
        /// <param name="targetValues">Expected values at boundary</param>
        /// <returns>Boundary residual</returns>
        Matrix<T> ComputeBoundaryLoss(
            Matrix<T> boundaryPoints,
            Matrix<T> predictions,
            Matrix<T> targetValues);
    }
}
```

### Automatic Differentiation Layer

```csharp
namespace AiDotNet.PhysicsInformed.AutoDiff
{
    /// <summary>
    /// Provides automatic differentiation capabilities for computing derivatives
    /// of network outputs with respect to inputs.
    /// </summary>
    /// <typeparam name="T">Data type</typeparam>
    public interface IAutoDiffLayer<T> where T : struct
    {
        /// <summary>
        /// Computes first-order derivatives ∂u/∂x using automatic differentiation.
        /// </summary>
        /// <param name="network">The neural network</param>
        /// <param name="inputs">Input points [batch, inputDim]</param>
        /// <param name="inputIndex">Index of input variable to differentiate</param>
        /// <returns>First derivatives [batch, outputDim]</returns>
        Matrix<T> ComputeFirstDerivative(
            IPhysicsInformedNetwork<T> network,
            Matrix<T> inputs,
            int inputIndex);

        /// <summary>
        /// Computes second-order derivatives ∂²u/∂x² using automatic differentiation.
        /// </summary>
        Matrix<T> ComputeSecondDerivative(
            IPhysicsInformedNetwork<T> network,
            Matrix<T> inputs,
            int inputIndex);

        /// <summary>
        /// Computes mixed derivatives ∂²u/∂x∂t.
        /// </summary>
        Matrix<T> ComputeMixedDerivative(
            IPhysicsInformedNetwork<T> network,
            Matrix<T> inputs,
            int inputIndex1,
            int inputIndex2);
    }
}
```

### Collocation Point Sampling

```csharp
namespace AiDotNet.PhysicsInformed.Sampling
{
    /// <summary>
    /// Generates collocation points where physics constraints are enforced.
    /// </summary>
    /// <typeparam name="T">Data type</typeparam>
    public interface ICollocationSampler<T> where T : struct
    {
        /// <summary>
        /// Samples points uniformly in the domain.
        /// </summary>
        /// <param name="numPoints">Number of points to sample</param>
        /// <param name="bounds">Domain bounds [inputDim, 2] (min/max for each dimension)</param>
        /// <returns>Sampled points [numPoints, inputDim]</returns>
        Matrix<T> SampleUniform(int numPoints, Matrix<T> bounds);

        /// <summary>
        /// Samples points using Latin Hypercube Sampling for better coverage.
        /// </summary>
        Matrix<T> SampleLatinHypercube(int numPoints, Matrix<T> bounds);

        /// <summary>
        /// Samples boundary points for enforcing boundary conditions.
        /// </summary>
        Matrix<T> SampleBoundary(int numPoints, Matrix<T> bounds, BoundaryType type);
    }

    public enum BoundaryType
    {
        Initial,    // t=0
        Left,       // x=x_min
        Right,      // x=x_max
        Top,        // y=y_max
        Bottom      // y=y_min
    }
}
```

---

## Implementation Steps

### Step 1: Implement Automatic Differentiation

**File**: `src/PhysicsInformed/AutoDiff/AutoDiffEngine.cs`

```csharp
public class AutoDiffEngine<T> : IAutoDiffLayer<T> where T : struct
{
    private readonly T _epsilon; // Small perturbation for finite differences

    public AutoDiffEngine(T epsilon)
    {
        _epsilon = epsilon;
    }

    public Matrix<T> ComputeFirstDerivative(
        IPhysicsInformedNetwork<T> network,
        Matrix<T> inputs,
        int inputIndex)
    {
        // Approach 1: Finite differences (simple but less accurate)
        // f'(x) ≈ [f(x+ε) - f(x-ε)] / (2ε)

        var batchSize = inputs.Rows;
        var inputDim = inputs.Columns;

        // Create perturbed inputs: x + ε * e_i
        var inputsPlus = inputs.Clone();
        var inputsMinus = inputs.Clone();

        for (int i = 0; i < batchSize; i++)
        {
            inputsPlus[i, inputIndex] = Add(inputs[i, inputIndex], _epsilon);
            inputsMinus[i, inputIndex] = Subtract(inputs[i, inputIndex], _epsilon);
        }

        // Forward pass with perturbed inputs
        var outputsPlus = network.Forward(inputsPlus);
        var outputsMinus = network.Forward(inputsMinus);

        // Compute finite difference: [f(x+ε) - f(x-ε)] / (2ε)
        var derivatives = new Matrix<T>(batchSize, outputsPlus.Columns);
        var twoEpsilon = Multiply(_epsilon, Convert.ChangeType(2, typeof(T)));

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < derivatives.Columns; j++)
            {
                var diff = Subtract(outputsPlus[i, j], outputsMinus[i, j]);
                derivatives[i, j] = Divide(diff, twoEpsilon);
            }
        }

        return derivatives;
    }

    public Matrix<T> ComputeSecondDerivative(
        IPhysicsInformedNetwork<T> network,
        Matrix<T> inputs,
        int inputIndex)
    {
        // Second derivative: f''(x) ≈ [f(x+ε) - 2f(x) + f(x-ε)] / ε²

        var batchSize = inputs.Rows;
        var inputsPlus = inputs.Clone();
        var inputsMinus = inputs.Clone();

        for (int i = 0; i < batchSize; i++)
        {
            inputsPlus[i, inputIndex] = Add(inputs[i, inputIndex], _epsilon);
            inputsMinus[i, inputIndex] = Subtract(inputs[i, inputIndex], _epsilon);
        }

        var outputsCenter = network.Forward(inputs);
        var outputsPlus = network.Forward(inputsPlus);
        var outputsMinus = network.Forward(inputsMinus);

        var derivatives = new Matrix<T>(batchSize, outputsCenter.Columns);
        var epsilonSquared = Multiply(_epsilon, _epsilon);

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < derivatives.Columns; j++)
            {
                // f''(x) = [f(x+ε) - 2f(x) + f(x-ε)] / ε²
                var numerator = Add(
                    Subtract(outputsPlus[i, j], Multiply(outputsCenter[i, j], 2.0)),
                    outputsMinus[i, j]);
                derivatives[i, j] = Divide(numerator, epsilonSquared);
            }
        }

        return derivatives;
    }

    // Helper methods for generic arithmetic
    private T Add(T a, T b) => (dynamic)a + (dynamic)b;
    private T Subtract(T a, T b) => (dynamic)a - (dynamic)b;
    private T Multiply(T a, T b) => (dynamic)a * (dynamic)b;
    private T Divide(T a, T b) => (dynamic)a / (dynamic)b;
}
```

**Key Learning Points**:
- Finite differences approximate derivatives numerically
- First derivative uses centered difference: `[f(x+ε) - f(x-ε)] / (2ε)`
- Second derivative: `[f(x+ε) - 2f(x) + f(x-ε)] / ε²`
- Trade-off: Smaller ε gives more accurate derivatives but risks numerical instability

**Alternative: Reverse-Mode AD** (Advanced)
For production, use reverse-mode automatic differentiation (like PyTorch's autograd) which is more efficient and accurate than finite differences.

---

### Step 2: Implement Physics Equations

**File**: `src/PhysicsInformed/Equations/HeatEquation.cs`

```csharp
public class HeatEquation<T> : IPhysicsEquation<T> where T : struct
{
    private readonly T _alpha; // Thermal diffusivity

    public HeatEquation(T alpha)
    {
        _alpha = alpha;
    }

    public List<DerivativeSpec> RequiredDerivatives => new List<DerivativeSpec>
    {
        new DerivativeSpec { Name = "du/dt", InputIndex = 1, Order = 1 },   // Time derivative
        new DerivativeSpec { Name = "d2u/dx2", InputIndex = 0, Order = 2 }  // Spatial second derivative
    };

    public Matrix<T> ComputeResidual(
        Matrix<T> inputs,
        Matrix<T> outputs,
        Dictionary<string, Matrix<T>> derivatives)
    {
        // Heat equation: ∂u/∂t - α ∂²u/∂x² = 0
        // Residual = ∂u/∂t - α ∂²u/∂x² (should be zero)

        var dudt = derivatives["du/dt"];
        var d2udx2 = derivatives["d2u/dx2"];

        var batchSize = inputs.Rows;
        var residuals = new Matrix<T>(batchSize, 1);

        for (int i = 0; i < batchSize; i++)
        {
            // residual = dudt - alpha * d2udx2
            residuals[i, 0] = Subtract(
                dudt[i, 0],
                Multiply(_alpha, d2udx2[i, 0]));
        }

        return residuals;
    }

    private T Subtract(T a, T b) => (dynamic)a - (dynamic)b;
    private T Multiply(T a, T b) => (dynamic)a * (dynamic)b;
}
```

**More Physics Equations to Implement**:

1. **Burgers' Equation** (nonlinear PDE):
   ```
   ∂u/∂t + u ∂u/∂x - ν ∂²u/∂x² = 0
   ```

2. **Wave Equation**:
   ```
   ∂²u/∂t² - c² ∂²u/∂x² = 0
   ```

3. **Navier-Stokes** (fluid dynamics):
   ```
   ∂u/∂t + (u·∇)u = -∇p + ν∇²u
   ```

---

### Step 3: Implement Collocation Sampling

**File**: `src/PhysicsInformed/Sampling/CollocationSampler.cs`

```csharp
public class CollocationSampler<T> : ICollocationSampler<T> where T : struct
{
    private readonly Random _random;

    public CollocationSampler(int seed = 42)
    {
        _random = new Random(seed);
    }

    public Matrix<T> SampleUniform(int numPoints, Matrix<T> bounds)
    {
        // bounds: [inputDim, 2] where bounds[i,0]=min, bounds[i,1]=max
        var inputDim = bounds.Rows;
        var samples = new Matrix<T>(numPoints, inputDim);

        for (int i = 0; i < numPoints; i++)
        {
            for (int j = 0; j < inputDim; j++)
            {
                var min = Convert.ToDouble(bounds[j, 0]);
                var max = Convert.ToDouble(bounds[j, 1]);
                var value = min + _random.NextDouble() * (max - min);
                samples[i, j] = (T)Convert.ChangeType(value, typeof(T));
            }
        }

        return samples;
    }

    public Matrix<T> SampleLatinHypercube(int numPoints, Matrix<T> bounds)
    {
        // Latin Hypercube Sampling ensures better coverage
        var inputDim = bounds.Rows;
        var samples = new Matrix<T>(numPoints, inputDim);

        for (int j = 0; j < inputDim; j++)
        {
            // Divide dimension into numPoints intervals
            var intervals = Enumerable.Range(0, numPoints).ToList();
            Shuffle(intervals); // Randomize order

            var min = Convert.ToDouble(bounds[j, 0]);
            var max = Convert.ToDouble(bounds[j, 1]);
            var intervalSize = (max - min) / numPoints;

            for (int i = 0; i < numPoints; i++)
            {
                var intervalStart = min + intervals[i] * intervalSize;
                var value = intervalStart + _random.NextDouble() * intervalSize;
                samples[i, j] = (T)Convert.ChangeType(value, typeof(T));
            }
        }

        return samples;
    }

    public Matrix<T> SampleBoundary(int numPoints, Matrix<T> bounds, BoundaryType type)
    {
        var inputDim = bounds.Rows;

        switch (type)
        {
            case BoundaryType.Initial: // t=0
                var initialPoints = SampleUniform(numPoints,
                    GetReducedBounds(bounds, excludeDim: 1));
                // Set t=0 for all points
                return SetDimension(initialPoints, 1, bounds[1, 0]);

            case BoundaryType.Left: // x=x_min
                var leftPoints = SampleUniform(numPoints,
                    GetReducedBounds(bounds, excludeDim: 0));
                return SetDimension(leftPoints, 0, bounds[0, 0]);

            // Similar for Right, Top, Bottom...

            default:
                throw new ArgumentException($"Unknown boundary type: {type}");
        }
    }

    private void Shuffle<T>(List<T> list)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            var temp = list[i];
            list[i] = list[j];
            list[j] = temp;
        }
    }
}
```

---

### Step 4: Implement PINN Loss Function

**File**: `src/PhysicsInformed/Loss/PINNLoss.cs`

```csharp
public class PINNLoss<T> : ILossFunction<T> where T : struct
{
    private readonly IPhysicsEquation<T> _equation;
    private readonly IBoundaryCondition<T> _boundary;
    private readonly IAutoDiffLayer<T> _autoDiff;

    // Loss weights
    private readonly T _dataWeight;
    private readonly T _physicsWeight;
    private readonly T _boundaryWeight;

    public PINNLoss(
        IPhysicsEquation<T> equation,
        IBoundaryCondition<T> boundary,
        IAutoDiffLayer<T> autoDiff,
        T dataWeight,
        T physicsWeight,
        T boundaryWeight)
    {
        _equation = equation;
        _boundary = boundary;
        _autoDiff = autoDiff;
        _dataWeight = dataWeight;
        _physicsWeight = physicsWeight;
        _boundaryWeight = boundaryWeight;
    }

    public T Compute(
        IPhysicsInformedNetwork<T> network,
        Matrix<T> dataPoints,
        Matrix<T> dataLabels,
        Matrix<T> collocationPoints,
        Matrix<T> boundaryPoints,
        Matrix<T> boundaryValues)
    {
        // 1. Data Loss: MSE on labeled data
        var predictions = network.Forward(dataPoints);
        var dataLoss = MeanSquaredError(predictions, dataLabels);

        // 2. Physics Loss: Residual of differential equation
        var physicsLoss = ComputePhysicsLoss(network, collocationPoints);

        // 3. Boundary Loss: Enforce boundary conditions
        var boundaryLoss = ComputeBoundaryLoss(network, boundaryPoints, boundaryValues);

        // 4. Weighted combination
        var totalLoss = Add(
            Multiply(_dataWeight, dataLoss),
            Add(
                Multiply(_physicsWeight, physicsLoss),
                Multiply(_boundaryWeight, boundaryLoss)));

        return totalLoss;
    }

    private T ComputePhysicsLoss(
        IPhysicsInformedNetwork<T> network,
        Matrix<T> collocationPoints)
    {
        // Compute all required derivatives
        var derivatives = new Dictionary<string, Matrix<T>>();

        foreach (var spec in _equation.RequiredDerivatives)
        {
            if (spec.Order == 1)
            {
                derivatives[spec.Name] = _autoDiff.ComputeFirstDerivative(
                    network, collocationPoints, spec.InputIndex);
            }
            else if (spec.Order == 2)
            {
                derivatives[spec.Name] = _autoDiff.ComputeSecondDerivative(
                    network, collocationPoints, spec.InputIndex);
            }
        }

        // Compute physics residual
        var outputs = network.Forward(collocationPoints);
        var residuals = _equation.ComputeResidual(collocationPoints, outputs, derivatives);

        // MSE of residuals (should be zero)
        return MeanSquared(residuals);
    }

    private T ComputeBoundaryLoss(
        IPhysicsInformedNetwork<T> network,
        Matrix<T> boundaryPoints,
        Matrix<T> boundaryValues)
    {
        var predictions = network.Forward(boundaryPoints);
        return _boundary.ComputeBoundaryLoss(boundaryPoints, predictions, boundaryValues);
    }

    private T MeanSquaredError(Matrix<T> predictions, Matrix<T> targets)
    {
        var sum = default(T);
        var count = predictions.Rows * predictions.Columns;

        for (int i = 0; i < predictions.Rows; i++)
        {
            for (int j = 0; j < predictions.Columns; j++)
            {
                var error = Subtract(predictions[i, j], targets[i, j]);
                sum = Add(sum, Multiply(error, error));
            }
        }

        return Divide(sum, Convert.ChangeType(count, typeof(T)));
    }
}
```

**Key Learning Points**:
- Three components: data loss (supervised), physics loss (PDE residual), boundary loss (initial/boundary conditions)
- Loss weights control the trade-off between fitting data and satisfying physics
- Physics loss measures how much the network violates the differential equation
- Typical weights: data=1.0, physics=0.1-1.0, boundary=0.1-1.0

---

### Step 5: Implement PINN Network

**File**: `src/PhysicsInformed/Networks/PhysicsInformedNN.cs`

```csharp
public class PhysicsInformedNN<T> : IPhysicsInformedNetwork<T> where T : struct
{
    private readonly List<ILayer<T>> _layers;
    private readonly IAutoDiffLayer<T> _autoDiff;

    public PhysicsInformedNN(List<int> layerSizes, IAutoDiffLayer<T> autoDiff)
    {
        _autoDiff = autoDiff;
        _layers = new List<ILayer<T>>();

        // Build fully connected network
        for (int i = 0; i < layerSizes.Count - 1; i++)
        {
            _layers.Add(new DenseLayer<T>(layerSizes[i], layerSizes[i + 1]));

            // Add activation (except for output layer)
            if (i < layerSizes.Count - 2)
            {
                _layers.Add(new TanhActivation<T>()); // Tanh works well for PINNs
            }
        }
    }

    public Matrix<T> Forward(Matrix<T> inputs)
    {
        var output = inputs;
        foreach (var layer in _layers)
        {
            output = layer.Forward(output);
        }
        return output;
    }

    public Matrix<T> ComputeDerivatives(Matrix<T> inputs, int order, int inputIndex)
    {
        if (order == 1)
        {
            return _autoDiff.ComputeFirstDerivative(this, inputs, inputIndex);
        }
        else if (order == 2)
        {
            return _autoDiff.ComputeSecondDerivative(this, inputs, inputIndex);
        }
        else
        {
            throw new ArgumentException($"Derivative order {order} not supported");
        }
    }

    public Matrix<T> ComputePhysicsResidual(Matrix<T> collocationPoints)
    {
        // This is handled by the loss function
        throw new NotImplementedException("Use PINNLoss.ComputePhysicsLoss instead");
    }
}
```

**Network Architecture Tips**:
- **Depth**: 4-8 hidden layers work well for most PDEs
- **Width**: 20-50 neurons per layer
- **Activation**: Tanh or Swish (smooth activations help with derivatives)
- **Initialization**: Xavier/Glorot initialization
- **Avoid ReLU**: Non-smooth activations cause issues with second derivatives

---

### Step 6: Training Loop

**File**: `src/PhysicsInformed/Training/PINNTrainer.cs`

```csharp
public class PINNTrainer<T> where T : struct
{
    private readonly IPhysicsInformedNetwork<T> _network;
    private readonly PINNLoss<T> _lossFunction;
    private readonly IOptimizer<T> _optimizer;
    private readonly ICollocationSampler<T> _sampler;

    public PINNTrainer(
        IPhysicsInformedNetwork<T> network,
        PINNLoss<T> lossFunction,
        IOptimizer<T> optimizer,
        ICollocationSampler<T> sampler)
    {
        _network = network;
        _lossFunction = lossFunction;
        _optimizer = optimizer;
        _sampler = sampler;
    }

    public TrainingResults<T> Train(
        Matrix<T> dataPoints,
        Matrix<T> dataLabels,
        Matrix<T> bounds,
        int numCollocationPoints,
        int numBoundaryPoints,
        int epochs)
    {
        var losses = new List<T>();

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // 1. Sample new collocation points each epoch
            var collocationPoints = _sampler.SampleLatinHypercube(
                numCollocationPoints, bounds);

            // 2. Sample boundary points
            var boundaryPoints = _sampler.SampleBoundary(
                numBoundaryPoints, bounds, BoundaryType.Initial);
            var boundaryValues = GetInitialCondition(boundaryPoints);

            // 3. Compute loss
            var loss = _lossFunction.Compute(
                _network,
                dataPoints,
                dataLabels,
                collocationPoints,
                boundaryPoints,
                boundaryValues);

            // 4. Backpropagation and update
            _optimizer.Step(_network, loss);

            losses.Add(loss);

            if (epoch % 100 == 0)
            {
                Console.WriteLine($"Epoch {epoch}: Loss = {loss}");
            }
        }

        return new TrainingResults<T> { Losses = losses };
    }

    private Matrix<T> GetInitialCondition(Matrix<T> points)
    {
        // Define initial condition u(x, t=0)
        // Example: Gaussian u(x,0) = exp(-x²)
        var values = new Matrix<T>(points.Rows, 1);
        for (int i = 0; i < points.Rows; i++)
        {
            var x = Convert.ToDouble(points[i, 0]);
            values[i, 0] = (T)Convert.ChangeType(Math.Exp(-x * x), typeof(T));
        }
        return values;
    }
}
```

**Training Tips**:
1. **Collocation Points**: Resample each epoch for better coverage
2. **Learning Rate**: Start with 1e-3, reduce if loss plateaus
3. **Epochs**: 10k-50k epochs typical for PDEs
4. **Adaptive Weights**: Adjust loss weights if one component dominates
5. **Convergence**: Monitor both total loss and individual components

---

## Testing Strategy

### Unit Tests

**File**: `tests/PhysicsInformed/HeatEquationTests.cs`

```csharp
[TestClass]
public class HeatEquationPINNTests
{
    [TestMethod]
    public void TestHeatEquation_AnalyticalSolution()
    {
        // Analytical solution: u(x,t) = exp(-α π² t) * sin(π x)
        // This satisfies heat equation with α and boundary conditions u(0,t)=u(1,t)=0

        var alpha = 0.01;
        var equation = new HeatEquation<double>(alpha);
        var autoDiff = new AutoDiffEngine<double>(1e-5);
        var network = new PhysicsInformedNN<double>(
            new List<int> { 2, 32, 32, 32, 1 }, autoDiff);

        // Generate training data from analytical solution
        var dataPoints = GenerateGridPoints(50, 50); // 50x50 grid in (x,t)
        var dataLabels = ComputeAnalyticalSolution(dataPoints, alpha);

        // Train PINN
        var loss = new PINNLoss<double>(equation, ..., autoDiff, 1.0, 0.1, 0.1);
        var optimizer = new AdamOptimizer<double>(learningRate: 0.001);
        var sampler = new CollocationSampler<double>();
        var trainer = new PINNTrainer<double>(network, loss, optimizer, sampler);

        var bounds = new Matrix<double>(2, 2);
        bounds[0, 0] = 0.0; bounds[0, 1] = 1.0; // x in [0,1]
        bounds[1, 0] = 0.0; bounds[1, 1] = 1.0; // t in [0,1]

        var results = trainer.Train(
            dataPoints, dataLabels, bounds,
            numCollocationPoints: 1000,
            numBoundaryPoints: 100,
            epochs: 10000);

        // Verify final loss is small
        var finalLoss = results.Losses.Last();
        Assert.IsTrue(finalLoss < 0.01, $"Loss too high: {finalLoss}");

        // Verify predictions match analytical solution
        var testPoints = GenerateGridPoints(20, 20);
        var predictions = network.Forward(testPoints);
        var expected = ComputeAnalyticalSolution(testPoints, alpha);

        var mse = ComputeMSE(predictions, expected);
        Assert.IsTrue(mse < 0.001, $"MSE too high: {mse}");
    }

    [TestMethod]
    public void TestAutoDiff_FirstDerivative()
    {
        // Test that ∂(x²)/∂x = 2x
        var autoDiff = new AutoDiffEngine<double>(1e-5);
        var network = new SimpleSquareNetwork(); // f(x) = x²

        var inputs = new Matrix<double>(1, 1);
        inputs[0, 0] = 3.0; // Test at x=3

        var derivative = autoDiff.ComputeFirstDerivative(network, inputs, 0);

        // Expected: 2*3 = 6
        Assert.AreEqual(6.0, derivative[0, 0], 0.001);
    }

    [TestMethod]
    public void TestPhysicsResidual_ZeroForAnalyticalSolution()
    {
        // If we feed the analytical solution to the physics equation,
        // residual should be zero

        var alpha = 0.01;
        var equation = new HeatEquation<double>(alpha);
        var autoDiff = new AutoDiffEngine<double>(1e-5);

        var network = new AnalyticalHeatNetwork(alpha); // Returns exp(-α π² t) sin(π x)

        var testPoints = GenerateGridPoints(10, 10);
        var outputs = network.Forward(testPoints);

        var derivatives = new Dictionary<string, Matrix<double>>
        {
            ["du/dt"] = autoDiff.ComputeFirstDerivative(network, testPoints, 1),
            ["d2u/dx2"] = autoDiff.ComputeSecondDerivative(network, testPoints, 0)
        };

        var residuals = equation.ComputeResidual(testPoints, outputs, derivatives);

        // Residual should be near zero
        var maxResidual = residuals.Data.Max(Math.Abs);
        Assert.IsTrue(maxResidual < 0.01, $"Max residual: {maxResidual}");
    }
}
```

### Integration Tests

Test complete workflow:
1. Generate synthetic PDE data
2. Train PINN
3. Verify physics constraints are satisfied
4. Compare to analytical solution (if available)

---

## Common Pitfalls

### 1. Poor Loss Weighting

**Problem**: One loss component dominates, network ignores others

**Solution**:
- Monitor individual loss components during training
- Use adaptive weighting: increase weight of slow-improving components
- Typical ranges: data=1.0, physics=0.01-1.0, boundary=0.1-1.0

### 2. Insufficient Collocation Points

**Problem**: Physics only enforced at sparse points, violations between

**Solution**:
- Use 1000+ collocation points for 1D problems
- Use Latin Hypercube Sampling for better coverage
- Resample each epoch to explore domain

### 3. Numerical Instability in Derivatives

**Problem**: Finite differences with wrong epsilon cause gradient explosion

**Solution**:
- Use epsilon = 1e-4 to 1e-6 (not too small!)
- Consider reverse-mode AD for production
- Use double precision for derivative computation

### 4. Stiff Equations

**Problem**: Large variations in solution scale cause training instability

**Solution**:
- Normalize inputs to [-1, 1]
- Use adaptive learning rates
- Try curriculum learning: easier problems first

### 5. Activation Function Choice

**Problem**: ReLU has zero second derivative, breaks second-order PDEs

**Solution**:
- Use Tanh, Swish, or Sine activations (smooth and differentiable)
- Avoid ReLU for problems requiring second derivatives

---

## Advanced Topics

### 1. Inverse Problems

Learn unknown parameters in PDEs from data:
```
Given: Data u(x,t)
Find: α in ∂u/∂t = α ∂²u/∂x²
```

Treat α as learnable parameter, optimize jointly with network weights.

### 2. Adaptive Collocation

Resample more points where residual is high:
```
Sample new points with probability ∝ |residual|
```

### 3. Transfer Learning

Pre-train on simple PDE, fine-tune on complex variant:
```
1. Train on heat equation with α=0.01
2. Fine-tune on heat equation with α=0.1
```

### 4. Multi-Fidelity Learning

Combine cheap low-fidelity simulations with expensive high-fidelity data.

### 5. Causality and Temporal Training

For time-dependent PDEs, train sequentially in time:
```
1. Train on t ∈ [0, 0.1]
2. Use as initial condition for t ∈ [0.1, 0.2]
```

---

## Performance Optimization

### 1. Batch Processing

Process collocation points in batches to leverage parallelism.

### 2. GPU Acceleration

Use GPU for matrix operations (forward pass and derivatives).

### 3. Automatic Differentiation

Replace finite differences with reverse-mode AD (10-100x faster).

### 4. Adaptive Sampling

Focus computational effort on high-error regions.

---

## Validation and Verification

### Checklist

- [ ] Derivatives computed correctly (test with known functions)
- [ ] Physics residual near zero at collocation points
- [ ] Boundary conditions satisfied (residual < 1e-3)
- [ ] Predictions match analytical solution (if available)
- [ ] Loss converges smoothly (no oscillations)
- [ ] Inference time < 10ms for 1000 points

### Benchmark Problems

1. **1D Heat Equation**: Start here, has analytical solution
2. **Burgers' Equation**: Nonlinear, test robustness
3. **2D Poisson Equation**: Test higher dimensions
4. **Navier-Stokes**: Ultimate test (complex, multi-variable)

---

## Resources

### Papers
- Raissi et al., "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations" (2019)
- Raissi et al., "Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations" (2020)

### Code Examples
- DeepXDE: Python library for PINNs (great reference implementation)
- PyTorch PINNs examples

### Mathematical Background
- Review PDEs, finite differences, and calculus
- Understand automatic differentiation principles

---

## Success Metrics

### Functionality
- [ ] Can solve 1D heat equation
- [ ] Physics residual < 1e-3
- [ ] Matches analytical solution with MSE < 0.001

### Code Quality
- [ ] Unit tests for derivatives
- [ ] Integration tests with known PDEs
- [ ] Modular design (equation, network, loss separate)

### Performance
- [ ] Training completes in < 5 minutes for 1D problem
- [ ] Inference < 10ms for 1000 points

### Documentation
- [ ] API documentation with XML comments
- [ ] Example notebook showing heat equation solution
- [ ] Guide for adding new PDEs

---

## Next Steps

After completing PINNs:
1. Explore **Neural Operators** (e.g., FNO) for faster inference
2. Study **DeepONet** for learning solution operators
3. Apply to real scientific problems (climate, materials, fluids)
4. Investigate **uncertainty quantification** in PINNs

**Congratulations!** You've learned how to combine deep learning with physics, a powerful paradigm for scientific computing.
