# Execution Engine Prototype Plan

## Objective
Validate Gemini's execution engine approach with a minimal prototype before rolling out to entire library.

## Prototype Scope

### What We're Testing
1. ✅ Execution engine pattern (IEngine, CpuEngine, GpuEngine)
2. ✅ Vector delegation to engine
3. ✅ Optimizer refactoring (vectorized operations)
4. ✅ GPU dispatch for float types
5. ✅ CPU fallback for non-float types
6. ✅ Performance comparison (CPU vs GPU)

### What We're NOT Changing
- ❌ Existing Vector/Matrix/Tensor classes (prototype uses copies)
- ❌ Existing optimizers (prototype creates new ones)
- ❌ Any production code
- ❌ Public APIs

## Implementation Steps

### Phase 1: Core Engine Infrastructure (2-3 hours)

#### Step 1.1: Create IEngine Interface
**File:** `src/Engines/IEngine.cs`

```csharp
namespace AiDotNet.Engines;

/// <summary>
/// Execution engine for mathematical operations.
/// Implementations can target CPU, GPU, or other accelerators.
/// </summary>
public interface IEngine
{
    // Basic vector operations needed for Adam optimizer
    Vector<T> Add<T>(Vector<T> a, Vector<T> b);
    Vector<T> Subtract<T>(Vector<T> a, Vector<T> b);
    Vector<T> Multiply<T>(Vector<T> a, Vector<T> b);  // Element-wise
    Vector<T> Multiply<T>(Vector<T> a, T scalar);
    Vector<T> Divide<T>(Vector<T> a, Vector<T> b);    // Element-wise
    Vector<T> Divide<T>(Vector<T> a, T scalar);
    Vector<T> Sqrt<T>(Vector<T> a);
    Vector<T> Power<T>(Vector<T> a, T exponent);

    // Metadata
    string Name { get; }
    bool SupportsGpu { get; }
}
```

#### Step 1.2: Create AiDotNetEngine Static Class
**File:** `src/Engines/AiDotNetEngine.cs`

```csharp
namespace AiDotNet.Engines;

public static class AiDotNetEngine
{
    private static IEngine _current = new CpuEngine();

    public static IEngine Current
    {
        get => _current;
        set => _current = value ?? throw new ArgumentNullException(nameof(value));
    }

    // Auto-detection helper
    public static void AutoDetectAndConfigureGpu()
    {
        try
        {
            var gpuEngine = new GpuEngine();
            if (gpuEngine.SupportsGpu)
            {
                Current = gpuEngine;
                Console.WriteLine($"[AiDotNet] GPU acceleration enabled: {gpuEngine.Name}");
            }
        }
        catch
        {
            Console.WriteLine("[AiDotNet] No GPU detected, using CPU");
        }
    }
}
```

#### Step 1.3: Implement CpuEngine (Basic)
**File:** `src/Engines/CpuEngine.cs`

```csharp
namespace AiDotNet.Engines;

public class CpuEngine : IEngine
{
    public string Name => "CPU Engine";
    public bool SupportsGpu => false;

    public Vector<T> Add<T>(Vector<T> a, Vector<T> b)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(a.Length);

        for (int i = 0; i < a.Length; i++)
        {
            result[i] = numOps.Add(a[i], b[i]);
        }

        return result;
    }

    // Implement other operations similarly...
}
```

#### Step 1.4: Implement GpuEngine (Float Only)
**File:** `src/Engines/GpuEngine.cs`

```csharp
using ILGPU;
using ILGPU.Runtime;

namespace AiDotNet.Engines;

public class GpuEngine : IEngine
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;

    public string Name => $"GPU Engine ({_accelerator?.Name ?? "Not Available"})";
    public bool SupportsGpu => _accelerator != null;

    public GpuEngine()
    {
        try
        {
            _context = Context.CreateDefault();
            _accelerator = _context.GetPreferredDevice(preferCPU: false)
                .CreateAccelerator(_context);
        }
        catch
        {
            // GPU not available
        }
    }

    public Vector<T> Add<T>(Vector<T> a, Vector<T> b)
    {
        // Runtime type check
        if (typeof(T) == typeof(float) && SupportsGpu)
        {
            return AddGpu(a as Vector<float>, b as Vector<float>) as Vector<T>;
        }

        // Fallback to CPU for non-float types
        return new CpuEngine().Add(a, b);
    }

    private Vector<float> AddGpu(Vector<float> a, Vector<float> b)
    {
        var result = new Vector<float>(a.Length);

        using var gpuA = _accelerator.Allocate1D<float>(a.Length);
        using var gpuB = _accelerator.Allocate1D<float>(b.Length);
        using var gpuResult = _accelerator.Allocate1D<float>(a.Length);

        gpuA.CopyFromCPU(a.ToArray());
        gpuB.CopyFromCPU(b.ToArray());

        var kernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
            (index, aView, bView, resultView) =>
                resultView[index] = aView[index] + bView[index]);

        kernel(a.Length, gpuA.View, gpuB.View, gpuResult.View);
        _accelerator.Synchronize();

        gpuResult.CopyToCPU(result.ToArray());

        return result;
    }

    // Implement other GPU operations...
}
```

**Test:** Create simple test to verify engines work

```csharp
[Fact]
public void Engine_CpuEngine_AddVectors()
{
    var engine = new CpuEngine();
    var a = new Vector<float>(new float[] { 1, 2, 3 });
    var b = new Vector<float>(new float[] { 4, 5, 6 });

    var result = engine.Add(a, b);

    Assert.Equal(new float[] { 5, 7, 9 }, result.ToArray());
}

[Fact]
public void Engine_GpuEngine_AddVectors_Float()
{
    var engine = new GpuEngine();
    if (!engine.SupportsGpu) return; // Skip if no GPU

    var a = new Vector<float>(new float[] { 1, 2, 3 });
    var b = new Vector<float>(new float[] { 4, 5, 6 });

    var result = engine.Add(a, b);

    Assert.Equal(new float[] { 5, 7, 9 }, result.ToArray());
}
```

### Phase 2: Prototype Vector with Engine Delegation (1-2 hours)

#### Step 2.1: Create PrototypeVector
**File:** `src/Prototype/PrototypeVector.cs`

```csharp
namespace AiDotNet.Prototype;

/// <summary>
/// Prototype Vector that delegates operations to execution engine.
/// </summary>
public class PrototypeVector<T>
{
    private readonly T[] _data;

    public int Length => _data.Length;
    public T this[int index]
    {
        get => _data[index];
        set => _data[index] = value;
    }

    public PrototypeVector(int length)
    {
        _data = new T[length];
    }

    public PrototypeVector(T[] data)
    {
        _data = (T[])data.Clone();
    }

    // Operations delegate to current engine
    public PrototypeVector<T> Add(PrototypeVector<T> other)
    {
        // Convert to regular Vector for engine
        var thisVec = new Vector<T>(_data);
        var otherVec = new Vector<T>(other._data);

        var result = AiDotNetEngine.Current.Add(thisVec, otherVec);

        return new PrototypeVector<T>(result.ToArray());
    }

    public PrototypeVector<T> Multiply(T scalar)
    {
        var thisVec = new Vector<T>(_data);
        var result = AiDotNetEngine.Current.Multiply(thisVec, scalar);
        return new PrototypeVector<T>(result.ToArray());
    }

    // Add other operations as needed...

    public T[] ToArray() => (T[])_data.Clone();
}
```

**Test:** Verify delegation works

```csharp
[Fact]
public void PrototypeVector_UsesCurrentEngine()
{
    AiDotNetEngine.Current = new CpuEngine();

    var a = new PrototypeVector<float>(new float[] { 1, 2, 3 });
    var b = new PrototypeVector<float>(new float[] { 4, 5, 6 });

    var result = a.Add(b);

    Assert.Equal(new float[] { 5, 7, 9 }, result.ToArray());
}
```

### Phase 3: Prototype Adam Optimizer (2-3 hours)

#### Step 3.1: Create PrototypeAdamOptimizer
**File:** `src/Prototype/PrototypeAdamOptimizer.cs`

```csharp
namespace AiDotNet.Prototype;

public class PrototypeAdamOptimizer<T>
{
    private PrototypeVector<T>? _m;
    private PrototypeVector<T>? _v;
    private int _t;

    private readonly double _beta1 = 0.9;
    private readonly double _beta2 = 0.999;
    private readonly double _epsilon = 1e-8;
    private readonly T _learningRate;

    private readonly INumericOperations<T> _numOps;

    public PrototypeAdamOptimizer(T learningRate)
    {
        _learningRate = learningRate;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public PrototypeVector<T> UpdateParameters(
        PrototypeVector<T> parameters,
        PrototypeVector<T> gradients)
    {
        // Initialize on first call
        if (_m == null || _m.Length != parameters.Length)
        {
            _m = new PrototypeVector<T>(parameters.Length);
            _v = new PrototypeVector<T>(parameters.Length);
            _t = 0;
        }

        _t++;

        var one = _numOps.One;
        var beta1 = _numOps.FromDouble(_beta1);
        var beta2 = _numOps.FromDouble(_beta2);
        var epsilon = _numOps.FromDouble(_epsilon);

        // *** VECTORIZED OPERATIONS - NO FOR LOOPS! ***

        // m = beta1 * m + (1 - beta1) * gradient
        var oneMinusBeta1 = _numOps.Subtract(one, beta1);
        _m = _m.Multiply(beta1).Add(gradients.Multiply(oneMinusBeta1));

        // v = beta2 * v + (1 - beta2) * gradient^2
        var oneMinusBeta2 = _numOps.Subtract(one, beta2);
        var gradSquared = gradients.Multiply(gradients);
        _v = _v.Multiply(beta2).Add(gradSquared.Multiply(oneMinusBeta2));

        // Bias correction
        var beta1t = _numOps.Power(beta1, _numOps.FromDouble(_t));
        var mHat = _m.Divide(_numOps.Subtract(one, beta1t));

        var beta2t = _numOps.Power(beta2, _numOps.FromDouble(_t));
        var vHat = _v.Divide(_numOps.Subtract(one, beta2t));

        // Update = learningRate * mHat / (sqrt(vHat) + epsilon)
        var vHatSqrt = vHat.Sqrt();
        var denominator = vHatSqrt.Add(epsilon);
        var update = mHat.Divide(denominator).Multiply(_learningRate);

        // parameters = parameters - update
        return parameters.Subtract(update);
    }
}
```

**Test:** Verify optimizer works with CPU and GPU engines

```csharp
[Theory]
[InlineData(typeof(CpuEngine))]
[InlineData(typeof(GpuEngine))]
public void PrototypeAdamOptimizer_UpdatesParameters(Type engineType)
{
    // Setup engine
    var engine = (IEngine)Activator.CreateInstance(engineType);
    if (engineType == typeof(GpuEngine) && !((GpuEngine)engine).SupportsGpu)
        return; // Skip if no GPU

    AiDotNetEngine.Current = engine;

    // Create optimizer
    var optimizer = new PrototypeAdamOptimizer<float>(0.001f);

    // Test data
    var parameters = new PrototypeVector<float>(new float[] { 1, 2, 3 });
    var gradients = new PrototypeVector<float>(new float[] { 0.1f, 0.2f, 0.3f });

    // Update
    var updated = optimizer.UpdateParameters(parameters, gradients);

    // Parameters should have moved opposite to gradient
    Assert.True(updated[0] < parameters[0]);
    Assert.True(updated[1] < parameters[1]);
    Assert.True(updated[2] < parameters[2]);
}
```

### Phase 4: Simple Test Models (2-3 hours)

#### Step 4.1: Simple Neural Network
**File:** `src/Prototype/SimpleNeuralNetwork.cs`

```csharp
namespace AiDotNet.Prototype;

public class SimpleNeuralNetwork<T>
{
    private PrototypeVector<T> _weights;
    private PrototypeVector<T> _biases;
    private readonly PrototypeAdamOptimizer<T> _optimizer;

    public SimpleNeuralNetwork(int inputSize, int outputSize, T learningRate)
    {
        _weights = InitializeRandom(inputSize * outputSize);
        _biases = InitializeRandom(outputSize);
        _optimizer = new PrototypeAdamOptimizer<T>(learningRate);
    }

    public PrototypeVector<T> Forward(PrototypeVector<T> input)
    {
        // Simple linear: output = weights * input + biases
        // (This is simplified; real NN would have proper matrix ops)
        var output = MatVecMultiply(_weights, input).Add(_biases);
        return output;
    }

    public void Train(PrototypeVector<T> input, PrototypeVector<T> target)
    {
        // Forward pass
        var output = Forward(input);

        // Compute gradient (simplified MSE)
        var error = output.Subtract(target);
        var gradient = ComputeGradient(input, error);

        // Update with optimizer
        _weights = _optimizer.UpdateParameters(_weights, gradient);
    }

    // ... helper methods ...
}
```

#### Step 4.2: Simple Linear Regression
**File:** `src/Prototype/SimpleLinearRegression.cs`

```csharp
namespace AiDotNet.Prototype;

public class SimpleLinearRegression<T>
{
    private PrototypeVector<T> _coefficients;
    private readonly PrototypeAdamOptimizer<T> _optimizer;

    public SimpleLinearRegression(int featureCount, T learningRate)
    {
        _coefficients = InitializeRandom(featureCount);
        _optimizer = new PrototypeAdamOptimizer<T>(learningRate);
    }

    public T Predict(PrototypeVector<T> features)
    {
        // y = w^T x
        return DotProduct(_coefficients, features);
    }

    public void Train(PrototypeVector<T> features, T target)
    {
        // Prediction
        var prediction = Predict(features);

        // Gradient
        var error = prediction - target;
        var gradient = features.Multiply(error);

        // Update
        _coefficients = _optimizer.UpdateParameters(_coefficients, gradient);
    }
}
```

### Phase 5: Integration Tests (2 hours)

#### Test 5.1: Neural Network with Float (GPU)
```csharp
[Fact]
public void Integration_NeuralNetwork_Float_UsesGpu()
{
    AiDotNetEngine.Current = new GpuEngine();
    if (!((GpuEngine)AiDotNetEngine.Current).SupportsGpu)
        return;

    var nn = new SimpleNeuralNetwork<float>(10, 5, 0.001f);

    // Train for a few iterations
    for (int i = 0; i < 100; i++)
    {
        var input = RandomVector<float>(10);
        var target = RandomVector<float>(5);
        nn.Train(input, target);
    }

    // Verify it worked (loss should decrease)
    Assert.True(/* check loss decreased */);
}
```

#### Test 5.2: Neural Network with Double (CPU Fallback)
```csharp
[Fact]
public void Integration_NeuralNetwork_Double_FallsBackToCpu()
{
    AiDotNetEngine.Current = new GpuEngine();

    var nn = new SimpleNeuralNetwork<double>(10, 5, 0.001);

    // Should work but use CPU (double not supported on GPU yet)
    for (int i = 0; i < 100; i++)
    {
        var input = RandomVector<double>(10);
        var target = RandomVector<double>(5);
        nn.Train(input, target);
    }

    Assert.True(/* check it worked */);
}
```

#### Test 5.3: Linear Regression with Decimal (CPU Only)
```csharp
[Fact]
public void Integration_LinearRegression_Decimal_UsesCpu()
{
    AiDotNetEngine.Current = new GpuEngine();

    var lr = new SimpleLinearRegression<decimal>(5, 0.001m);

    // Decimal should work on CPU
    for (int i = 0; i < 100; i++)
    {
        var features = RandomVector<decimal>(5);
        var target = Random.Decimal();
        lr.Train(features, target);
    }

    Assert.True(/* check it worked */);
}
```

#### Test 5.4: Performance Benchmark
```csharp
[Fact]
public void Benchmark_CpuVsGpu_Float()
{
    var largeVector1 = RandomVector<float>(1_000_000);
    var largeVector2 = RandomVector<float>(1_000_000);

    // CPU benchmark
    AiDotNetEngine.Current = new CpuEngine();
    var cpuTime = Measure(() => largeVector1.Add(largeVector2));

    // GPU benchmark
    AiDotNetEngine.Current = new GpuEngine();
    if (!((GpuEngine)AiDotNetEngine.Current).SupportsGpu)
        return;

    var gpuTime = Measure(() => largeVector1.Add(largeVector2));

    Console.WriteLine($"CPU: {cpuTime}ms, GPU: {gpuTime}ms, Speedup: {cpuTime/gpuTime}x");

    // GPU should be faster for large vectors
    Assert.True(gpuTime < cpuTime);
}
```

## Success Criteria

Prototype is successful if:
- ✅ All tests pass with CpuEngine
- ✅ All tests pass with GpuEngine (for float)
- ✅ Non-float types fallback to CPU correctly
- ✅ GPU shows measurable speedup for large operations
- ✅ No changes to existing production code
- ✅ Optimizer refactoring (vectorized ops) is cleaner than current for-loops

## Next Steps After Successful Prototype

1. **Document findings** in prototype-results.md
2. **Create detailed GitHub issue** for full rollout
3. **Plan migration strategy** for existing optimizers
4. **Implement remaining IEngine operations**
5. **Add Matrix and Tensor support to engines**

## Regarding Tensor Standardization

**With execution engine approach, Tensor standardization for neural networks is OPTIONAL, not required.**

**Reasons:**
- Engine handles GPU dispatch regardless of internal type
- Layers can use Vector, Matrix, or Tensor - all work with engine
- GPU support is orthogonal to data structure choice

**However**, we may still WANT Tensor for neural networks for:
- Conceptual clarity (activations are ND tensors)
- Batch operations (easier with Tensor than Vector)
- Industry alignment (PyTorch/TensorFlow use tensors)

**Decision:** Test prototype first, then decide if Tensor standardization provides additional value.

## Timeline Estimate

- Phase 1 (Engine infrastructure): 2-3 hours
- Phase 2 (PrototypeVector): 1-2 hours
- Phase 3 (PrototypeAdam): 2-3 hours
- Phase 4 (Test models): 2-3 hours
- Phase 5 (Integration tests): 2 hours
- **Total: 9-13 hours** (1-2 days of focused work)

## Files to Create

1. `src/Engines/IEngine.cs`
2. `src/Engines/CpuEngine.cs`
3. `src/Engines/GpuEngine.cs`
4. `src/Engines/AiDotNetEngine.cs`
5. `src/Prototype/PrototypeVector.cs`
6. `src/Prototype/PrototypeAdamOptimizer.cs`
7. `src/Prototype/SimpleNeuralNetwork.cs`
8. `src/Prototype/SimpleLinearRegression.cs`
9. `tests/Prototype/EngineTests.cs`
10. `tests/Prototype/PrototypeIntegrationTests.cs`

## Risk Mitigation

- Prototype isolated in separate namespace - can't break production
- Tests validate each component independently
- Performance benchmarks verify GPU actually helps
- Easy to abandon if approach doesn't work

## Open Questions for Prototype

1. Is ILGPU the right GPU library, or should we try ComputeSharp?
2. What's the actual GPU vs CPU speedup for realistic workloads?
3. How much overhead does runtime type checking add?
4. Do we need double support on GPU, or is float sufficient?
5. Should we support multiple GPU engines (CUDA, OpenCL, etc.)?
