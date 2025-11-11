# Autodiff Layer Integration - Session Handoff

## Current Status (as of last commit)

**Branch:** `claude/autodiff-layer-integration-011CV1K5xX5dTYfGRXKXodCN`

### Completed Work

**TensorOperations Implemented:** 36 total
- Base operations (19): Add, Subtract, Multiply, Divide, MatMul, Transpose, Reshape, ReLU, Sigmoid, Tanh, ElementwiseMultiply, Sum, Mean, Variance, Exp, Log, Pow, Sqrt, Abs
- Session additions (17): Conv2D, ConvTranspose2D, MaxPool2D, AvgPool2D, Softmax, Concat, Pad, LayerNorm, BatchNorm, ReduceMax, ReduceMean, Split, Crop, Upsample, PixelShuffle, DilatedConv2D, DepthwiseConv2D

**Layers with Full Autodiff:** 22
1. DenseLayer
2. ActivationLayer
3. DropoutLayer
4. AddLayer
5. MultiplyLayer
6. ConvolutionalLayer
7. DeconvolutionalLayer
8. MaxPoolingLayer
9. PoolingLayer
10. BatchNormalizationLayer
11. LayerNormalizationLayer
12. AttentionLayer
13. GlobalPoolingLayer
14. GaussianNoiseLayer
15. MaskingLayer
16. SubpixelConvolutionalLayer
17. UpsamplingLayer
18. DepthwiseSeparableConvolutionalLayer
19. CroppingLayer
20. SplitLayer
21. DilatedConvolutionalLayer
22. SeparableConvolutionalLayer

### Remaining Work: 21 Layers

## HIGH PRIORITY: Production-Ready Layers (4 layers)

These layers are commonly used in production and need TensorOperations added:

### 1. LocallyConnectedLayer → LocallyConnectedConv2D operation
**File:** `src/NeuralNetworks/Layers/LocallyConnectedLayer.cs:???`
**Operation:** Locally connected (unshared convolution)
**Implementation Notes:**
- Like Conv2D but weights are NOT shared across spatial locations
- Each output position has its own unique set of weights
- Much larger parameter count than standard conv
- Forward: Different weights per spatial location
- Backward: Separate gradients for each location's weights

**Pseudo-code:**
```csharp
public static ComputationNode<T> LocallyConnectedConv2D(
    ComputationNode<T> input,
    ComputationNode<T> weights,  // [out_h, out_w, out_c, in_c, kH, kW]
    ComputationNode<T>? bias = null,
    int[]? stride = null,
    int[]? padding = null)
{
    // Each output spatial location has unique weights
    // weights is 6D instead of 4D!
}
```

### 2. SpatialTransformerLayer → AffineGrid + GridSample operations
**File:** `src/NeuralNetworks/Layers/SpatialTransformerLayer.cs:???`
**Operations:** Two-part operation
1. **AffineGrid**: Generate sampling grid from affine matrix
2. **GridSample**: Sample input using grid (bilinear interpolation)

**Implementation Notes:**
- Used for learnable spatial transformations
- Common in STNs (Spatial Transformer Networks)
- AffineGrid: Create meshgrid and apply affine transform
- GridSample: Bilinear interpolation with gradient support
- Both need careful gradient implementation

**Pseudo-code:**
```csharp
public static ComputationNode<T> AffineGrid(
    ComputationNode<T> theta,  // [batch, 2, 3] affine matrix
    int[] outputSize)  // [H, W]
{
    // Generate regular grid
    // Apply affine transform to grid points
    // Return transformed sampling coordinates
}

public static ComputationNode<T> GridSample(
    ComputationNode<T> input,
    ComputationNode<T> grid)  // sampling coordinates
{
    // Bilinear interpolation at grid points
    // Backward: gradients w.r.t both input and grid
}
```

### 3. RBFLayer → RBFKernel operation
**File:** `src/NeuralNetworks/Layers/RBFLayer.cs:???`
**Operation:** Radial Basis Function kernel
**Implementation Notes:**
- Compute RBF: `exp(-gamma * ||x - center||²)`
- Forward: Gaussian kernel centered at each RBF center
- Backward: Gradients for input, centers, and gamma

**Pseudo-code:**
```csharp
public static ComputationNode<T> RBFKernel(
    ComputationNode<T> input,  // [batch, features]
    ComputationNode<T> centers,  // [num_centers, features]
    ComputationNode<T> gamma)  // [num_centers]
{
    // For each center: compute distance to all inputs
    // Apply Gaussian: exp(-gamma * distance²)
    // Gradients flow through distance computation
}
```

### 4. LogVarianceLayer → Can use existing Log operation
**File:** `src/NeuralNetworks/Layers/LogVarianceLayer.cs:???`
**Status:** Likely can use existing operations
**Action Required:** Check if Variance exists, compose with Log
**Notes:** May just need `Log(Variance(input))` composition

## MEDIUM PRIORITY: Specialized Research Layers (17 layers)

These are research-oriented and require complex domain-specific implementations:

### Capsule Networks (3 layers)
- **CapsuleLayer** - Dynamic routing algorithm
- **DigitCapsuleLayer** - Digit capsule routing
- **PrimaryCapsuleLayer** - Primary capsule convolutions

**Operation Needed:** `DynamicRouting`
**Complexity:** ~300 lines
**Notes:**
- Iterative routing-by-agreement algorithm
- Coupling coefficients updated via softmax
- Multiple routing iterations
- Complex gradient through routing iterations

### Quantum Computing (2 layers)
- **QuantumLayer** - Quantum gate operations
- **MeasurementLayer** - Quantum measurement

**Operations Needed:** `QuantumGate`, `QuantumMeasurement`
**Complexity:** ~400 lines total
**Notes:**
- Complex number operations
- Unitary matrix constraints
- Measurement collapse
- May require complex number Tensor support

### Conditional Random Fields (1 layer)
- **ConditionalRandomFieldLayer** - CRF for structured prediction

**Operation Needed:** `ViterbiDecode`, `CRFForwardBackward`
**Complexity:** ~500 lines
**Notes:**
- Forward-backward algorithm
- Viterbi decoding
- Log-space computations for numerical stability
- Transition matrix gradients

### Graph Neural Networks (2 layers)
- **GraphConvolutionalLayer** - Graph convolution
- **SpatialPoolerLayer** - Hierarchical temporal memory

**Operations Needed:** `GraphConv`, `MessagePassing`
**Complexity:** ~300 lines
**Notes:**
- Adjacency matrix operations
- Message passing between nodes
- Aggregation functions
- Graph topology gradients

### Neuromorphic/Spiking (3 layers)
- **SpikingLayer** - Spiking neuron dynamics
- **SynapticPlasticityLayer** - STDP learning
- **TemporalMemoryLayer** - HTM temporal memory

**Operations Needed:** `SpikeDynamics`, `STDP`, `TemporalPooling`
**Complexity:** ~600 lines total
**Notes:**
- Temporal dynamics
- Spike-timing dependent plasticity
- Non-differentiable spikes (surrogate gradients)
- Complex state updates

### Other Specialized (6 layers)
- **RBMLayer** - Restricted Boltzmann Machine (needs `ContrastiveDivergence`)
- **AnomalyDetectorLayer** - Anomaly detection (domain-specific)
- **RepParameterizationLayer** - Reparameterization trick (may use existing ops)
- **ReadoutLayer** - Reservoir computing readout (may use existing ops)
- **DecoderLayer** - Sequence decoder (may use existing ops)
- **ExpertLayer** / **MixtureOfExpertsLayer** - Expert routing (may use existing ops)
- **ReconstructionLayer** - Autoencoder reconstruction (may use existing ops)

**Notes:** Some of these may be implementable with existing operations - needs investigation.

## Implementation Strategy

### Recommended Order

**Phase 1 (Completed):** Production Conv Variants ✅
1. ✅ Added `DilatedConv2D` operation
2. ✅ Updated DilatedConvolutionalLayer
3. ✅ Added `DepthwiseConv2D` operation
4. ✅ Updated DepthwiseSeparableConvolutionalLayer
5. ✅ Composed from DepthwiseConv2D + Conv2D
6. ✅ Updated SeparableConvolutionalLayer

**Phase 2 (Completed):** Simple Layers Using Existing Ops ✅
1. ✅ Updated CroppingLayer to use Crop
2. ✅ Updated SplitLayer to use Reshape (not Split - layer does reshape)
3. Investigate and update LogVarianceLayer, RepParameterizationLayer, ReadoutLayer

**Phase 3 (Next):** Advanced Production Ops
1. Add `LocallyConnectedConv2D` operation
2. Update LocallyConnectedLayer
3. Add `AffineGrid` + `GridSample` operations
4. Update SpatialTransformerLayer
5. Add `RBFKernel` operation
6. Update RBFLayer

**Phase 4:** Research Layers (Optional based on priority)
1. Capsule networks if needed by users
2. Graph neural networks if needed
3. Others as required

### Code Pattern for Adding TensorOperations

```csharp
/// <summary>
/// [Operation description]
/// </summary>
public static ComputationNode<T> OperationName(
    ComputationNode<T> input,
    /* other parameters */)
{
    var numOps = MathHelper.GetNumericOperations<T>();
    var inputShape = input.Value.Shape;

    // Validate inputs
    // ...

    // Compute output shape
    var outputShape = /* ... */;
    var result = new Tensor<T>(outputShape);

    // Forward pass implementation
    // ...

    void BackwardFunction(Tensor<T> gradient)
    {
        if (!input.RequiresGradient) return;

        if (input.Gradient == null)
            input.Gradient = new Tensor<T>(inputShape);

        // Backward pass implementation
        // ...
    }

    var node = new ComputationNode<T>(
        value: result,
        requiresGradient: input.RequiresGradient,
        parents: new List<ComputationNode<T>> { input },
        backwardFunction: BackwardFunction,
        name: null);

    var tape = GradientTape<T>.Current;
    if (tape != null && tape.IsRecording)
        tape.RecordOperation(node);

    return node;
}
```

### Code Pattern for Updating Layers

```csharp
private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
{
    if (_lastInput == null)
        throw new InvalidOperationException("Forward pass must be called before backward pass.");

    // Convert to computation nodes
    var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);

    // Apply TensorOperation
    var outputNode = Autodiff.TensorOperations<T>.OperationName(inputNode, /* params */);

    // Backward pass
    outputNode.Gradient = outputGradient;
    var topoOrder = GetTopologicalOrder(outputNode);
    for (int i = topoOrder.Count - 1; i >= 0; i--)
    {
        var node = topoOrder[i];
        if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
        {
            node.BackwardFunction(node.Gradient);
        }
    }

    return inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");
}

// Helper method (copy from other layers if needed)
private List<Autodiff.ComputationNode<T>> GetTopologicalOrder(Autodiff.ComputationNode<T> root)
{
    // Standard topological sort implementation
    // ...
}
```

## Testing

After implementing each operation:
1. Add test to `tests/AiDotNet.Tests/UnitTests/Autodiff/GradientCorrectnessTests.cs`
2. Compare autodiff gradients vs numerical gradients
3. Use tolerance of 1e-4 for float, 1e-6 for double

Example test structure:
```csharp
[Fact]
public void OperationName_AutodiffGradients_MatchNumericalGradients()
{
    // Setup
    var input = CreateTestTensor();
    var inputNode = TensorOperations<float>.Variable(input, "input", requiresGradient: true);

    // Forward
    var output = TensorOperations<float>.OperationName(inputNode, /* params */);

    // Backward
    var outputGrad = CreateTestGradient();
    output.Gradient = outputGrad;
    output.Backward();

    // Numerical gradient
    var numerical = ComputeNumericalGradient(input, /* ... */);

    // Assert
    AssertGradientsMatch(inputNode.Gradient, numerical, tolerance: 1e-4);
}
```

## Current Branch Status

```
branch: claude/autodiff-layer-integration-011CV1K5xX5dTYfGRXKXodCN
status: Clean, all changes committed and pushed
latest commit: "feat: Update UpsamplingLayer to use Upsample operation"
```

## Documentation to Update

After completing operations, update:
- `docs/AutodiffImplementation.md` - Update layer counts and operation list
- Add new operations to "TensorOperations Implemented" section
- Move completed layers from "Partial" to "Fully Implemented"

## Key Files

- TensorOperations: `src/Autodiff/TensorOperations.cs` (currently 3628 lines)
- Test file: `tests/AiDotNet.Tests/UnitTests/Autodiff/GradientCorrectnessTests.cs`
- Documentation: `docs/AutodiffImplementation.md`
- All layer files: `src/NeuralNetworks/Layers/*.cs`

## Notes

- All operations must be production-ready with correct gradients
- No simplified or approximate implementations
- Follow existing code style and documentation patterns
- Each operation should have comprehensive XML documentation
- Test gradients numerically for correctness

## Session Summary

**Previous sessions:** Added Split, Crop, Upsample, PixelShuffle, DilatedConv2D operations and updated SubpixelConvolutionalLayer, UpsamplingLayer, GaussianNoiseLayer, MaskingLayer.

**This session:** Added DepthwiseConv2D operation (250 lines) and updated 5 layers:
1. DepthwiseSeparableConvolutionalLayer - Uses DepthwiseConv2D + Conv2D
2. CroppingLayer - Uses Crop operation
3. SplitLayer - Uses Reshape operation
4. DilatedConvolutionalLayer - Uses DilatedConv2D operation
5. SeparableConvolutionalLayer - Composes DepthwiseConv2D + Conv2D

**Current status:** 22 layers with full autodiff (was 17), 36 TensorOperations (was 35), 21 layers remaining.

**Priority:** 4 high-priority production layers (LocallyConnected, SpatialTransformer, RBF, LogVariance) followed by 17 specialized research layers as needed.
