# Mixed-Precision Training Architecture

## Executive Summary

This document outlines the architecture for implementing mixed-precision training in AiDotNet. Mixed-precision training uses both FP16 (half precision) and FP32 (single precision) floating-point formats to accelerate training while maintaining model accuracy.

**Status**: Architecture design (not yet implemented)
**Estimated Implementation Time**: 20-40 hours
**Complexity Level**: High
**Dependencies**: CUDA/GPU support, FP16 numeric operations

## Background

### What is Mixed-Precision Training?

Mixed-precision training is a technique that uses lower-precision (FP16) numerical formats for some operations while keeping higher-precision (FP32) for others. This provides:

- **2-3x faster training** on modern GPUs with Tensor Cores
- **~50% reduction in memory usage**
- **Maintained or improved model accuracy** with proper techniques

### Industry Standard: NVIDIA's Approach

NVIDIA's mixed-precision training (used in PyTorch, TensorFlow) follows this pattern:

1. **FP16 Forward/Backward**: Most operations run in FP16 for speed
2. **FP32 Master Weights**: Parameter updates happen in FP32 for precision
3. **Loss Scaling**: Gradients are scaled to prevent underflow
4. **Dynamic Loss Scaling**: Automatically adjusts scale factor during training

## Proposed Architecture

### 1. Numeric Type System Extension

```csharp
// Extend INumericOperations to support precision conversion
public interface INumericOperations<T>
{
    // Existing methods...

    // NEW: Mixed-precision support
    TLower Cast<TLower>(T value);
    T CastFrom<TLower>(TLower value);
    int GetPrecisionBits();
}

// NEW: Precision modes
public enum PrecisionMode
{
    FP32,       // Full precision (default)
    FP16,       // Half precision
    Mixed,      // Mixed FP16/FP32
    BF16        // Brain float 16 (Google's format)
}
```

### 2. Mixed-Precision Training Context

```csharp
public class MixedPrecisionContext<THigh, TLow> : IDisposable
    where THigh : struct  // FP32
    where TLow : struct   // FP16
{
    /// <summary>
    /// Master copy of parameters in high precision.
    /// </summary>
    private Dictionary<string, Tensor<THigh>> _masterWeights;

    /// <summary>
    /// Working copy of parameters in low precision for forward/backward.
    /// </summary>
    private Dictionary<string, Tensor<TLow>> _workingWeights;

    /// <summary>
    /// Current loss scale factor.
    /// </summary>
    public double LossScale { get; private set; }

    /// <summary>
    /// Whether to use dynamic loss scaling.
    /// </summary>
    public bool DynamicScaling { get; set; }

    /// <summary>
    /// Converts master weights to working precision for forward pass.
    /// </summary>
    public void CastWeightsToLowPrecision();

    /// <summary>
    /// Scales gradients and casts back to high precision for update.
    /// </summary>
    public void UnscaleGradientsToHighPrecision();

    /// <summary>
    /// Adjusts loss scale based on gradient overflow/underflow detection.
    /// </summary>
    public void UpdateLossScale();
}
```

### 3. Integration with Tensor Operations

```csharp
// Add precision tracking to Tensor
public class Tensor<T>
{
    // NEW: Track precision mode
    public PrecisionMode Precision { get; internal set; }

    // NEW: Cast to different precision
    public Tensor<TOut> Cast<TOut>()
    {
        // Convert tensor to different numeric type
        // E.g., Tensor<float> -> Tensor<Half>
    }
}

// Update TensorOperations to handle mixed precision
public static class TensorOperations<T>
{
    // Automatic precision casting when operations mix types
    public static ComputationNode<T> Add(
        ComputationNode<T> a,
        ComputationNode<T> b,
        PrecisionMode? forcePrecision = null)
    {
        // If forcePrecision specified, cast inputs
        // Otherwise, use higher precision of the two inputs
    }
}
```

### 4. Loss Scaling Implementation

```csharp
public class LossScaler
{
    private double _scale;
    private int _growthInterval;
    private int _successfulSteps;
    private double _backoffFactor;
    private double _growthFactor;

    /// <summary>
    /// Scales the loss to prevent gradient underflow.
    /// </summary>
    public T ScaleLoss<T>(T loss)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return numOps.Multiply(loss, numOps.FromDouble(_scale));
    }

    /// <summary>
    /// Unscales gradients after backward pass.
    /// </summary>
    public void UnscaleGradients<T>(Dictionary<string, Tensor<T>> gradients)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inverseScale = numOps.FromDouble(1.0 / _scale);

        foreach (var grad in gradients.Values)
        {
            grad.Transform((x, _) => numOps.Multiply(x, inverseScale));
        }
    }

    /// <summary>
    /// Checks for overflow/underflow and adjusts scale dynamically.
    /// </summary>
    public bool CheckGradientsAndUpdateScale<T>(Dictionary<string, Tensor<T>> gradients)
    {
        // Check for NaN or Inf
        bool hasOverflow = DetectOverflow(gradients);

        if (hasOverflow)
        {
            // Reduce scale factor
            _scale *= _backoffFactor;
            _successfulSteps = 0;
            return false; // Skip this update
        }
        else
        {
            // Increase scale after successful steps
            _successfulSteps++;
            if (_successfulSteps >= _growthInterval)
            {
                _scale *= _growthFactor;
                _successfulSteps = 0;
            }
            return true; // Update is valid
        }
    }
}
```

### 5. Layer and Network Integration

```csharp
public abstract class LayerBase<T>
{
    // NEW: Precision mode for this layer
    public PrecisionMode ComputePrecision { get; set; }

    // Override in subclasses
    public virtual Tensor<T> ForwardMixedPrecision(Tensor<T> input)
    {
        // Cast input to layer's compute precision
        // Perform computation
        // Cast output back to expected precision
    }
}

public abstract class NeuralNetworkBase<T>
{
    // NEW: Mixed-precision training context
    protected MixedPrecisionContext<float, Half>? _mixedPrecisionContext;

    // NEW: Enable mixed-precision training
    public void EnableMixedPrecision(
        bool dynamicScaling = true,
        double initialScale = 65536.0)
    {
        _mixedPrecisionContext = new MixedPrecisionContext<float, Half>
        {
            DynamicScaling = dynamicScaling,
            LossScale = initialScale
        };
    }
}
```

## Implementation Phases

### Phase 1: Foundation (8-12 hours)
1. Add `Half` (FP16) support to `INumericOperations`
2. Implement precision casting in `MathHelper`
3. Add `Cast<TOut>()` method to `Tensor<T>`
4. Create unit tests for precision conversions

### Phase 2: Loss Scaling (4-6 hours)
1. Implement `LossScaler` class
2. Add overflow/underflow detection
3. Implement dynamic loss scaling algorithm
4. Test scaling with synthetic gradients

### Phase 3: Context Management (6-8 hours)
1. Implement `MixedPrecisionContext`
2. Master weights management (FP32 copy)
3. Working weights management (FP16 copy)
4. Gradient accumulation in FP32

### Phase 4: Network Integration (6-10 hours)
1. Update `NeuralNetworkBase` for mixed-precision
2. Modify training loop to use scaled loss
3. Integrate with optimizer updates
4. Add `EnableMixedPrecision()` API

### Phase 5: Validation & Optimization (4-8 hours)
1. Test on real models (ResNet, Transformer)
2. Verify accuracy is maintained
3. Benchmark performance improvements
4. Document best practices

## Technical Challenges

### Challenge 1: C# Half Type Limitations
**Problem**: C# `System.Half` (FP16) doesn't support all operations
**Solution**: Implement custom `HalfNumericOperations` with explicit casts

### Challenge 2: GPU Acceleration Required
**Problem**: Mixed-precision benefits require GPU Tensor Cores
**Solution**: Document as "CPU implementation for compatibility, GPU for performance"

### Challenge 3: Generic Constraints
**Problem**: C# generics don't allow `where T : float or Half`
**Solution**: Use runtime type checks or separate implementations

### Challenge 4: Gradient Underflow
**Problem**: FP16 range is [6e-8, 65504], small gradients underflow
**Solution**: Loss scaling (multiply loss by 2^16, then unscale gradients)

## API Design

### User-Facing API

```csharp
// Simple usage - enable with defaults
var model = new ResidualNeuralNetwork<float>(architecture);
model.EnableMixedPrecision();

// Advanced usage - configure scaling
model.EnableMixedPrecision(
    dynamicScaling: true,
    initialScale: 32768.0,
    scalingGrowthInterval: 2000,
    scalingBackoffFactor: 0.5);

// Train normally - mixed precision handled automatically
var result = await new PredictionModelBuilder<float, Matrix<float>, Vector<float>>()
    .ConfigureModel(model)
    .BuildAsync(trainingData, labels);
```

### Implementation Example

```csharp
public class MixedPrecisionTrainingLoop<T>
{
    public void TrainStep(
        Tensor<T> input,
        Tensor<T> target,
        MixedPrecisionContext context,
        LossScaler scaler)
    {
        // 1. Cast weights to FP16
        context.CastWeightsToLowPrecision();

        // 2. Forward pass in FP16
        var output = model.Forward(input.Cast<Half>());

        // 3. Compute loss in FP32
        var loss = lossFunction.Compute(
            output.Cast<float>(),
            target);

        // 4. Scale loss
        var scaledLoss = scaler.ScaleLoss(loss);

        // 5. Backward pass (gradients in FP16)
        var gradients = model.Backward(scaledLoss);

        // 6. Unscale and cast gradients to FP32
        context.UnscaleGradientsToHighPrecision(gradients);

        // 7. Check for overflow and update scale
        if (scaler.CheckGradientsAndUpdateScale(gradients))
        {
            // 8. Update master weights in FP32
            optimizer.UpdateParameters(context.MasterWeights, gradients);
        }
        else
        {
            // Skip this update due to gradient overflow
            Console.WriteLine("Gradient overflow detected, skipping update");
        }
    }
}
```

## Testing Strategy

### Unit Tests
- Precision conversion accuracy
- Loss scaling/unscaling correctness
- Overflow detection
- Dynamic scale adjustment

### Integration Tests
- Train simple model with mixed precision
- Compare convergence with FP32 baseline
- Verify final accuracy within 0.1% of FP32

### Performance Benchmarks
- Memory usage comparison
- Training speed comparison (GPU required)
- Forward/backward pass timing

## Best Practices (Future Documentation)

### When to Use Mixed-Precision
✅ **Good candidates:**
- Large models (>100M parameters)
- GPU training with Tensor Cores (V100, A100, RTX 3000+)
- Memory-constrained scenarios
- Long training runs

❌ **Not recommended:**
- CPU-only training (minimal benefit)
- Very small models (<1M parameters)
- When using custom layers with numerical instability

### Debugging Mixed-Precision Issues

**Symptom**: Loss becomes NaN
**Solutions:**
- Increase initial loss scale
- Use dynamic scaling
- Check for operations that produce large values (exp, pow)

**Symptom**: Slower than FP32
**Solutions:**
- Ensure GPU has Tensor Cores
- Check that operations actually run in FP16
- Verify batch size is large enough

**Symptom**: Lower final accuracy
**Solutions**:
- Keep batch normalization in FP32
- Keep loss computation in FP32
- Use larger learning rate

## References

### Academic Papers
- Micikevicius et al. (2017) "Mixed Precision Training" (NVIDIA)
- Kalamkar et al. (2019) "A Study of BFLOAT16 for Deep Learning Training"

### Industry Implementations
- PyTorch AMP: https://pytorch.org/docs/stable/amp.html
- TensorFlow Mixed Precision: https://www.tensorflow.org/guide/mixed_precision
- NVIDIA Apex: https://github.com/NVIDIA/apex

### Relevant Standards
- IEEE 754-2008: Half-precision floating-point format
- NVIDIA Tensor Core Programming Guide

## Future Enhancements

### BF16 Support (Brain Float 16)
- Same range as FP32, less precision
- Better for training than FP16 in some cases
- Requires CPU/GPU support

### INT8 Quantization
- Even faster than FP16
- Primarily for inference, not training
- Requires quantization-aware training

### Automatic Mixed Precision (AMP)
- Automatically determine which ops should be FP16 vs FP32
- Similar to PyTorch AMP
- Requires op-level profiling and heuristics

## Conclusion

Mixed-precision training is a significant feature that can provide substantial performance benefits. The architecture outlined here provides a roadmap for implementation while maintaining the clean, user-friendly API that AiDotNet is known for.

**Next Steps:**
1. Implement `Half` support in `INumericOperations`
2. Create `LossScaler` prototype
3. Validate approach with simple model
4. Incrementally integrate with existing codebase

This feature should be implemented as a separate epic/milestone given its complexity and the need for GPU infrastructure for proper validation.
