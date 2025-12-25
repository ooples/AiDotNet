# Golden Standard Patterns for AiDotNet Models and Layers

This document defines the production-ready patterns that ALL models and layers in the AiDotNet library must follow. Consistency is critical for maintainability and user experience.

## Table of Contents

1. [Layer Pattern](#layer-pattern)
2. [Model Pattern](#model-pattern)
3. [Common Conventions](#common-conventions)
4. [Documentation Standards](#documentation-standards)
5. [Interface Implementation Checklist](#interface-implementation-checklist)

---

## Layer Pattern

All layers MUST extend `LayerBase<T>` and follow these patterns:

### Class Declaration

```csharp
namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Brief description of what this layer does.
/// </summary>
/// <remarks>
/// <para>
/// Technical explanation of the layer's operation.
/// </para>
/// <para><b>For Beginners:</b> Simple analogy explanation.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ExampleLayer<T> : LayerBase<T>, IOptionalInterface<T>
{
    // Implementation
}
```

### Field Naming

```csharp
// Private fields use underscore prefix
private Tensor<T> _weights;
private Tensor<T> _biases;
private Tensor<T> _lastInput;    // Cached for backward pass
private Tensor<T> _lastOutput;   // Cached for backward pass
private Tensor<T>? _weightsGradient;  // Nullable for gradients
private Tensor<T>? _biasesGradient;
private readonly Random _random;

// Public properties use PascalCase
public int InputChannels { get; private set; }
public int OutputChannels { get; private set; }
```

### Constructor Pattern

```csharp
/// <summary>
/// Initializes a new instance with the specified parameters.
/// </summary>
/// <param name="inputDim">Description of parameter.</param>
/// <param name="outputDim">Description of parameter.</param>
/// <param name="activation">The activation function to apply. Defaults to ReLU if not specified.</param>
public ExampleLayer(
    int inputDim,
    int outputDim,
    IActivationFunction<T>? activation = null)
    : base(
        CalculateInputShape(inputDim),
        CalculateOutputShape(outputDim),
        activation ?? new ReLUActivation<T>())
{
    // Store configuration
    InputChannels = inputDim;
    OutputChannels = outputDim;

    // Initialize tensors
    _weights = new Tensor<T>([outputDim, inputDim]);
    _biases = new Tensor<T>([outputDim]);
    _lastInput = new Tensor<T>([inputDim]);
    _lastOutput = new Tensor<T>([outputDim]);

    // Use RandomHelper, NEVER new Random() directly
    _random = RandomHelper.CreateSecureRandom();

    // Initialize weights
    InitializeWeights();
}
```

### Required Abstract Members

```csharp
/// <summary>
/// Gets a value indicating whether this layer supports training.
/// </summary>
public override bool SupportsTraining => true;

/// <summary>
/// Gets whether this layer supports JIT compilation.
/// </summary>
public override bool SupportsJitCompilation => CanActivationBeJitted();

public override Tensor<T> Forward(Tensor<T> input)
{
    _lastInput = input;

    // Perform layer operation
    var output = /* computation */;

    // Apply activation
    _lastOutput = ApplyActivation(output);
    return _lastOutput;
}

public override Tensor<T> Backward(Tensor<T> outputGradient)
{
    // Apply activation derivative
    var delta = ApplyActivationDerivative(_lastOutput, outputGradient);

    // Compute gradients for weights and biases
    _weightsGradient = /* gradient computation */;
    _biasesGradient = /* gradient computation */;

    // Compute input gradient
    return /* input gradient */;
}

public override void UpdateParameters(T learningRate)
{
    // Update weights: w = w - lr * gradient
    // Use NumOps for type-safe operations
    for (int i = 0; i < _weights.Length; i++)
    {
        var update = NumOps.Multiply(learningRate, _weightsGradient[i]);
        _weights[i] = NumOps.Subtract(_weights[i], update);
    }
}

public override Vector<T> GetParameters()
{
    int totalParams = _weights.Length + _biases.Length;
    var parameters = new Vector<T>(totalParams);

    int index = 0;
    // Copy all parameters to vector
    foreach (var w in _weights.AsSpan())
        parameters[index++] = w;
    foreach (var b in _biases.AsSpan())
        parameters[index++] = b;

    return parameters;
}

public override void SetParameters(Vector<T> parameters)
{
    if (parameters.Length != ParameterCount)
        throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}");

    int index = 0;
    // Restore all parameters from vector
}

public override void ResetState()
{
    _lastInput = new Tensor<T>(InputShape);
    _lastOutput = new Tensor<T>(OutputShape);
}

public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
{
    // Create computation graph for JIT compilation
}
```

### Serialization

```csharp
public override void Serialize(BinaryWriter writer)
{
    base.Serialize(writer);

    // Write configuration
    writer.Write(InputChannels);
    writer.Write(OutputChannels);

    // Write tensors (use NumOps.ToDouble for type conversion)
    foreach (var w in _weights.AsSpan())
        writer.Write(NumOps.ToDouble(w));
}

public override void Deserialize(BinaryReader reader)
{
    base.Deserialize(reader);

    // Read configuration
    InputChannels = reader.ReadInt32();
    OutputChannels = reader.ReadInt32();

    // Read tensors (use NumOps.FromDouble for type conversion)
    _weights = new Tensor<T>([OutputChannels, InputChannels]);
    var span = _weights.AsWritableSpan();
    for (int i = 0; i < span.Length; i++)
        span[i] = NumOps.FromDouble(reader.ReadDouble());
}
```

---

## Model Pattern

All models MUST implement `IFullModel<T, TInput, TOutput>` and follow these patterns:

### Class Declaration

```csharp
namespace AiDotNet.Diffusion;

/// <summary>
/// Brief description of the model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class ExampleModelBase<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    protected Random RandomGenerator;
    protected readonly ILossFunction<T> LossFunction;
}
```

### IFullModel Interface Requirements

```csharp
// IModel<TInput, TOutput, ModelMetadata<T>>
public virtual void Train(Tensor<T> input, Tensor<T> expectedOutput);
public virtual Tensor<T> Predict(Tensor<T> input);
public virtual ModelMetadata<T> GetModelMetadata();

// IModelSerializer
public virtual byte[] Serialize();
public virtual void Deserialize(byte[] data);
public virtual void SaveModel(string filePath);
public virtual void LoadModel(string filePath);

// ICheckpointableModel
public virtual void SaveState(Stream stream);
public virtual void LoadState(Stream stream);

// IParameterizable<T, TInput, TOutput>
public abstract Vector<T> GetParameters();
public abstract void SetParameters(Vector<T> parameters);
public virtual IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters);

// IFeatureAware
public virtual IEnumerable<int> GetActiveFeatureIndices();
public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices);
public virtual bool IsFeatureUsed(int featureIndex);

// IFeatureImportance<T>
public virtual Dictionary<string, T> GetFeatureImportance();

// ICloneable<IFullModel<T, TInput, TOutput>>
public abstract IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy();

// IGradientComputable<T, TInput, TOutput>
public virtual Vector<T> ComputeGradients(Tensor<T> input, Tensor<T> target, ILossFunction<T>? lossFunction = null);
public virtual void ApplyGradients(Vector<T> gradients, T learningRate);

// IJitCompilable<T>
public virtual bool SupportsJitCompilation => false;
public virtual ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes);

// IFullModel<T, TInput, TOutput>
public ILossFunction<T> DefaultLossFunction => LossFunction;
```

---

## Common Conventions

### Numeric Operations

```csharp
// ALWAYS use NumOps for type T operations
protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

// Convert literals to T
T one = NumOps.One;
T zero = NumOps.Zero;
T value = NumOps.FromDouble(0.5);

// Arithmetic operations
T result = NumOps.Add(a, b);
T product = NumOps.Multiply(a, b);
T quotient = NumOps.Divide(a, b);
T difference = NumOps.Subtract(a, b);
T sqrtVal = NumOps.Sqrt(x);

// Convert back to double
double d = NumOps.ToDouble(value);
```

### Random Number Generation

```csharp
// NEVER use new Random() or new Random(seed) directly
// ALWAYS use RandomHelper

// For reproducible results with a seed
var rng = seed.HasValue
    ? RandomHelper.CreateSeededRandom(seed.Value)
    : RandomHelper.CreateSecureRandom();

// For thread-safe random access
protected static Random Random => RandomHelper.ThreadSafeRandom;
```

### Tensor Operations

```csharp
// Read-only span access
var readSpan = tensor.AsSpan();

// Writable span access
var writeSpan = tensor.AsWritableSpan();

// NEVER use AsReadOnlySpan() - it doesn't exist!
```

### Nullable Parameters with Defaults

```csharp
// Constructor with nullable optional parameters
public ExampleLayer(
    int requiredParam,
    IActivationFunction<T>? activation = null,
    int? seed = null)
{
    // Apply defaults in constructor body
    _activation = activation ?? new ReLUActivation<T>();
    _random = seed.HasValue
        ? RandomHelper.CreateSeededRandom(seed.Value)
        : RandomHelper.CreateSecureRandom();
}
```

### Property Initialization

```csharp
// Use NumOps.FromDouble for generic type property defaults
public T SomeWeight { get; set; }

// In constructor:
SomeWeight = NumOps.FromDouble(0.01);

// NEVER use: SomeWeight = default!; (null-forgiving operator)
```

---

## Documentation Standards

### XML Documentation Template

```csharp
/// <summary>
/// One-line description of what this member does.
/// </summary>
/// <remarks>
/// <para>
/// Detailed technical explanation of how it works.
/// Include mathematical formulas where appropriate.
/// </para>
/// <para>
/// <b>For Beginners:</b> Simple explanation using analogies.
/// - Bullet points for key concepts
/// - Real-world examples
/// - What this means in practice
/// </para>
/// </remarks>
/// <param name="paramName">Description of the parameter.</param>
/// <returns>Description of the return value.</returns>
/// <exception cref="ArgumentException">When this exception is thrown.</exception>
```

### "For Beginners" Section Guidelines

1. Use simple analogies
2. Avoid jargon
3. Explain why, not just what
4. Include examples of real-world use cases
5. Keep it concise but informative

---

## Interface Implementation Checklist

### For Layers (LayerBase<T>)

- [ ] Extends `LayerBase<T>`
- [ ] Has underscore-prefixed private fields
- [ ] Uses `NumOps` for all numeric operations
- [ ] Uses `RandomHelper` for random number generation
- [ ] Implements `SupportsTraining` property
- [ ] Implements `SupportsJitCompilation` property
- [ ] Implements `Forward(Tensor<T>)`
- [ ] Implements `Backward(Tensor<T>)`
- [ ] Implements `UpdateParameters(T)`
- [ ] Implements `GetParameters()`
- [ ] Implements `SetParameters(Vector<T>)`
- [ ] Implements `ResetState()`
- [ ] Implements `ExportComputationGraph(List<ComputationNode<T>>)`
- [ ] Overrides `Serialize(BinaryWriter)` if additional state
- [ ] Overrides `Deserialize(BinaryReader)` if additional state
- [ ] Has comprehensive XML documentation with "For Beginners" sections

### For Models (IFullModel<T, TInput, TOutput>)

- [ ] Implements `IFullModel<T, Tensor<T>, Tensor<T>>`
- [ ] Has `protected static readonly INumericOperations<T> NumOps`
- [ ] Has `protected Random RandomGenerator`
- [ ] Has `protected readonly ILossFunction<T> LossFunction`
- [ ] Implements `DefaultLossFunction` property
- [ ] Implements all interface methods (see Model Pattern section)
- [ ] Uses `NumOps` for all numeric operations
- [ ] Uses `RandomHelper` for random number generation
- [ ] Has comprehensive XML documentation with "For Beginners" sections

---

## Examples

### Reference Layer: ConvolutionalLayer

Located at: `src/NeuralNetworks/Layers/ConvolutionalLayer.cs`

Key patterns demonstrated:
- Constructor with nullable activation parameter
- Static `Configure` factory methods
- Proper weight initialization
- Forward/Backward pass caching
- Serialization/Deserialization
- JIT compilation support

### Reference Base Class: NoisePredictorBase

Located at: `src/Diffusion/NoisePredictors/NoisePredictorBase.cs`

Key patterns demonstrated:
- IFullModel implementation
- INoisePredictor interface
- Timestep embedding computation
- Numerical gradient computation
- Stream-based serialization
