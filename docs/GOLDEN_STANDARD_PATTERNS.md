# Golden Standard Patterns for AiDotNet Models and Layers

This document defines the production-ready patterns that ALL models and layers in the AiDotNet library must follow. Consistency is critical for maintainability and user experience.

## Table of Contents

1. [Layer Pattern](#layer-pattern)
2. [Model Pattern](#model-pattern)
   - [Neural Network Model Pattern](#neural-network-model-pattern-primary---feedforwardneuralnetwork-as-golden-standard)
   - [Layer Creation Pattern](#layer-creation-pattern-critical)
   - [Standalone Model Pattern](#standalone-model-pattern-for-non-layer-based-models)
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

There are two main model patterns in AiDotNet:
1. **Neural Network Models** - Extend `NeuralNetworkBase<T>` (preferred for layer-based models)
2. **Standalone Models** - Implement `IFullModel<T, TInput, TOutput>` directly

### Neural Network Model Pattern (Primary - FeedForwardNeuralNetwork as Golden Standard)

All neural network models MUST extend `NeuralNetworkBase<T>`:

```csharp
namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Brief description of the neural network.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Technical explanation of the network architecture.
/// </para>
/// <para>
/// <b>For Beginners:</b> Simple analogy explaining what this network does.
/// </para>
/// </remarks>
public class ExampleNeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// The loss function used to calculate the error between predicted and expected outputs.
    /// </summary>
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// The optimization algorithm used to update the network's parameters during training.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
}
```

### Constructor Pattern (Neural Network)

```csharp
/// <summary>
/// Initializes a new instance with the specified architecture.
/// </summary>
/// <param name="architecture">The architecture defining the structure of the neural network.</param>
/// <param name="optimizer">The optimization algorithm. If null, Adam optimizer is used.</param>
/// <param name="lossFunction">The loss function. If null, selected based on task type.</param>
/// <param name="maxGradNorm">Maximum gradient norm for gradient clipping.</param>
public ExampleNeuralNetwork(
    NeuralNetworkArchitecture<T> architecture,
    IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
    ILossFunction<T>? lossFunction = null,
    double maxGradNorm = 1.0)
    : base(architecture,
           lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType),
           maxGradNorm)
{
    // Apply defaults for optional parameters
    _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
    _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

    // Initialize layers from architecture
    InitializeLayers();
}
```

### Layer Creation Pattern (CRITICAL)

**All neural network models MUST support both user-provided layers AND default industry-standard layers.**

The layer creation pattern uses a dual approach:
1. **User-Provided Layers**: If the user supplies layers via `Architecture.Layers`, use those
2. **Default Layers**: If no layers provided, use `LayerHelper<T>.CreateDefault<ModelType>Layers()` for industry-standard defaults

#### The InitializeLayers() Pattern

```csharp
/// <summary>
/// Initializes the layers of the neural network based on the architecture.
/// </summary>
/// <remarks>
/// <para>
/// This method follows the dual-approach pattern:
/// 1. If the user provides custom layers, use those (with validation)
/// 2. Otherwise, use LayerHelper to create industry-standard default layers
/// </para>
/// </remarks>
protected override void InitializeLayers()
{
    if (Architecture.Layers != null && Architecture.Layers.Count > 0)
    {
        // Use the layers provided by the user
        Layers.AddRange(Architecture.Layers);
        ValidateCustomLayers(Layers);
    }
    else
    {
        // Use default layer configuration if no layers are provided
        // Each model type has its own CreateDefault<ModelType>Layers method
        Layers.AddRange(LayerHelper<T>.CreateDefaultFeedForwardLayers(Architecture));
    }
}
```

#### LayerHelper Factory Methods

The `LayerHelper<T>` class is a static factory that creates industry-standard layer configurations:

```csharp
// Located at: src/Helpers/LayerHelper.cs
public static class LayerHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    // Feed-forward networks
    public static IEnumerable<ILayer<T>> CreateDefaultLayers(
        NeuralNetworkArchitecture<T> architecture,
        int hiddenLayerCount = 1,
        int hiddenLayerSize = 64,
        int outputSize = 1);

    // Alias for CreateDefaultLayers (feed-forward)
    public static IEnumerable<ILayer<T>> CreateDefaultFeedForwardLayers(
        NeuralNetworkArchitecture<T> architecture) =>
        CreateDefaultLayers(architecture);

    // Convolutional Neural Networks
    public static IEnumerable<ILayer<T>> CreateDefaultCNNLayers(
        NeuralNetworkArchitecture<T> architecture,
        int convLayerCount = 2,
        int filterCount = 32,
        int kernelSize = 3,
        int denseLayerCount = 1,
        int denseLayerSize = 64,
        int outputSize = 1);

    // ResNet architectures
    public static IEnumerable<ILayer<T>> CreateDefaultResNetLayers(
        NeuralNetworkArchitecture<T> architecture,
        int blockCount = 3,
        int blockSize = 2);

    // LSTM-based temporal networks
    public static IEnumerable<ILayer<T>> CreateDefaultOccupancyTemporalLayers(
        NeuralNetworkArchitecture<T> architecture,
        int historyWindowSize);

    // Deep Boltzmann Machines
    public static IEnumerable<ILayer<T>> CreateDefaultDeepBoltzmannMachineLayers(
        NeuralNetworkArchitecture<T> architecture);

    // Attention-based (Transformer) networks
    public static IEnumerable<ILayer<T>> CreateDefaultAttentionLayers(
        NeuralNetworkArchitecture<T> architecture);
}
```

#### Naming Convention for LayerHelper Methods

When adding a new model type, add a corresponding method to `LayerHelper<T>`:

```
CreateDefault<ModelName>Layers(NeuralNetworkArchitecture<T> architecture, [optional params])
```

Examples:
- `CreateDefaultFeedForwardLayers` → FeedForwardNeuralNetwork
- `CreateDefaultCNNLayers` → ConvolutionalNeuralNetwork
- `CreateDefaultResNetLayers` → ResNetNeuralNetwork
- `CreateDefaultTransformerLayers` → TransformerNeuralNetwork
- `CreateDefaultDiffusionLayers` → DiffusionNeuralNetwork (future)

#### Industry-Standard Defaults

Each LayerHelper method should use industry-standard defaults:

| Model Type | Default Activations | Default Sizes |
|------------|---------------------|---------------|
| Feed-Forward | ReLU (hidden), Softmax (output) | 64 neurons per hidden layer |
| CNN | ReLU + MaxPool | 32 filters, 3x3 kernel |
| ResNet | ReLU + BatchNorm | 64 initial channels |
| LSTM/Temporal | Tanh + Sigmoid (recurrent) | 64-32 hidden sizes |
| Attention | ReLU | 4-8 heads |

#### Complete Example: FeedForwardNeuralNetwork

```csharp
public class FeedForwardNeuralNetwork<T> : NeuralNetworkBase<T>
{
    public FeedForwardNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture,
               lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType),
               maxGradNorm)
    {
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

        // CRITICAL: Always call InitializeLayers() at end of constructor
        InitializeLayers();
    }

    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // User provided custom layers - use them
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // No layers provided - use LayerHelper for defaults
            Layers.AddRange(LayerHelper<T>.CreateDefaultFeedForwardLayers(Architecture));
        }
    }
}
```

### Required Override Methods (Neural Network)

```csharp
/// <summary>
/// Gets whether this network supports training.
/// </summary>
public override bool SupportsTraining => true;

/// <summary>
/// Initializes the layers of the neural network based on the architecture.
/// </summary>
protected override void InitializeLayers()
{
    if (Architecture.Layers != null && Architecture.Layers.Count > 0)
    {
        // Use user-provided layers
        Layers.AddRange(Architecture.Layers);
        ValidateCustomLayers(Layers);
    }
    else
    {
        // Use default layer configuration
        Layers.AddRange(LayerHelper<T>.CreateDefaultFeedForwardLayers(Architecture));
    }
}

/// <summary>
/// Makes a prediction using the neural network.
/// </summary>
public override Tensor<T> Predict(Tensor<T> input)
{
    IsTrainingMode = false;
    TensorValidator.ValidateShape(input, Architecture.GetInputShape(), nameof(ExampleNeuralNetwork<T>), "prediction");

    var predictions = Forward(input);

    IsTrainingMode = true;
    return predictions;
}

/// <summary>
/// Performs a forward pass through the network.
/// </summary>
public Tensor<T> Forward(Tensor<T> input)
{
    TensorValidator.ValidateShape(input, Architecture.GetInputShape(), nameof(ExampleNeuralNetwork<T>), "forward pass");

    Tensor<T> output = input;
    foreach (var layer in Layers)
    {
        output = layer.Forward(output);
    }
    return output;
}

/// <summary>
/// Performs a backward pass through the network.
/// </summary>
public Tensor<T> Backward(Tensor<T> outputGradient)
{
    Tensor<T> gradient = outputGradient;
    for (int i = Layers.Count - 1; i >= 0; i--)
    {
        gradient = Layers[i].Backward(gradient);
    }
    return gradient;
}

/// <summary>
/// Trains the neural network using the provided input and expected output.
/// </summary>
public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
{
    IsTrainingMode = true;

    // Forward pass
    var prediction = Forward(input);

    // Calculate loss (including auxiliary losses from layers)
    var primaryLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
    T auxiliaryLoss = NumOps.Zero;
    foreach (var auxLayer in Layers.OfType<IAuxiliaryLossLayer<T>>().Where(l => l.UseAuxiliaryLoss))
    {
        var weightedAuxLoss = NumOps.Multiply(auxLayer.ComputeAuxiliaryLoss(), auxLayer.AuxiliaryLossWeight);
        auxiliaryLoss = NumOps.Add(auxiliaryLoss, weightedAuxLoss);
    }
    LastLoss = NumOps.Add(primaryLoss, auxiliaryLoss);

    // Calculate gradient and backpropagate
    var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
    Backward(Tensor<T>.FromVector(outputGradient));

    // Update parameters
    _optimizer.UpdateParameters(Layers);

    IsTrainingMode = false;
}

/// <summary>
/// Retrieves metadata about the neural network model.
/// </summary>
public override ModelMetadata<T> GetModelMetadata()
{
    return new ModelMetadata<T>
    {
        ModelType = ModelType.FeedForwardNetwork,
        AdditionalInfo = new Dictionary<string, object>
        {
            { "InputShape", Architecture.GetInputShape() },
            { "OutputShape", Architecture.GetOutputShape() },
            { "LayerCount", Layers.Count },
            { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() },
            { "TaskType", Architecture.TaskType.ToString() },
            { "ParameterCount", GetParameterCount() }
        },
        ModelData = this.Serialize()
    };
}

/// <summary>
/// Creates a new instance with the same configuration.
/// </summary>
protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
{
    return new ExampleNeuralNetwork<T>(
        Architecture,
        _optimizer,
        _lossFunction,
        Convert.ToDouble(MaxGradNorm));
}
```

### Serialization (Neural Network)

```csharp
/// <summary>
/// Serializes network-specific data to a binary writer.
/// </summary>
protected override void SerializeNetworkSpecificData(BinaryWriter writer)
{
    writer.Write(_optimizer.GetType().FullName ?? "AdamOptimizer");
    writer.Write(_lossFunction.GetType().FullName ?? "MeanSquaredErrorLoss");
}

/// <summary>
/// Deserializes network-specific data from a binary reader.
/// </summary>
protected override void DeserializeNetworkSpecificData(BinaryReader reader)
{
    string optimizerType = reader.ReadString();
    string lossFunctionType = reader.ReadString();
    // Reconstruct optimizer and loss function from types if needed
}
```

---

### Standalone Model Pattern (For Non-Layer-Based Models)

For models that don't use the layer architecture (like diffusion base classes):

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

### IFullModel Interface Requirements (Standalone)

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

### For Neural Network Models (NeuralNetworkBase<T>)

- [ ] Extends `NeuralNetworkBase<T>`
- [ ] Has `private ILossFunction<T> _lossFunction`
- [ ] Has `private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer`
- [ ] Constructor accepts `NeuralNetworkArchitecture<T>` as first parameter
- [ ] Constructor has nullable `optimizer` and `lossFunction` parameters with defaults
- [ ] Calls `InitializeLayers()` at end of constructor
- [ ] Overrides `SupportsTraining` property
- [ ] Overrides `InitializeLayers()` method
- [ ] **InitializeLayers() uses dual approach:**
  - [ ] Checks `Architecture.Layers != null && Architecture.Layers.Count > 0`
  - [ ] If user provided layers: Uses `Layers.AddRange(Architecture.Layers)` and `ValidateCustomLayers(Layers)`
  - [ ] If no layers provided: Uses `LayerHelper<T>.CreateDefault<ModelType>Layers(Architecture)`
- [ ] **Has corresponding LayerHelper method** (e.g., `CreateDefaultMyModelLayers`)
- [ ] Overrides `Predict(Tensor<T>)` method
- [ ] Implements `Forward(Tensor<T>)` method
- [ ] Implements `Backward(Tensor<T>)` method
- [ ] Overrides `Train(Tensor<T>, Tensor<T>)` method
- [ ] Overrides `GetModelMetadata()` method
- [ ] Overrides `CreateNewInstance()` method
- [ ] Overrides `SerializeNetworkSpecificData(BinaryWriter)` method
- [ ] Overrides `DeserializeNetworkSpecificData(BinaryReader)` method
- [ ] Uses `NumOps` for all numeric operations
- [ ] Uses `RandomHelper.ThreadSafeRandom` for random numbers (via base class)
- [ ] Has comprehensive XML documentation with "For Beginners" sections

### For Standalone Models (IFullModel<T, TInput, TOutput>)

- [ ] Implements `IFullModel<T, Tensor<T>, Tensor<T>>`
- [ ] Has `protected static readonly INumericOperations<T> NumOps`
- [ ] Has `protected Random RandomGenerator`
- [ ] Has `protected readonly ILossFunction<T> LossFunction`
- [ ] Implements `DefaultLossFunction` property
- [ ] Implements all interface methods (see Standalone Model Pattern section)
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
- Proper weight initialization using `RandomHelper`
- Forward/Backward pass caching (`_lastInput`, `_lastOutput`)
- Gradient storage (`_kernelsGradient`, `_biasesGradient`)
- Serialization/Deserialization with `NumOps.ToDouble`/`FromDouble`
- JIT compilation support via `ExportComputationGraph`

### Reference Layer: TimeEmbeddingLayer

Located at: `src/NeuralNetworks/Layers/TimeEmbeddingLayer.cs`

Key patterns demonstrated:
- Diffusion-specific layer for timestep conditioning
- Two-layer MLP projection with SiLU activation
- Sinusoidal embedding computation
- Uses `RandomHelper.CreateSeededRandom()` for reproducibility
- Complete forward/backward implementation

### Reference Neural Network Model: FeedForwardNeuralNetwork

Located at: `src/NeuralNetworks/FeedForwardNeuralNetwork.cs`

Key patterns demonstrated:
- Extends `NeuralNetworkBase<T>` (THE GOLDEN STANDARD)
- Constructor with nullable optimizer and loss function
- Proper layer initialization from architecture using dual approach
- Forward/Backward pass implementation
- Training loop with auxiliary loss support
- Model metadata generation
- Network-specific serialization

### Reference Helper: LayerHelper

Located at: `src/Helpers/LayerHelper.cs`

Key patterns demonstrated:
- Static factory class for creating default layer configurations
- Uses `IEnumerable<ILayer<T>>` with `yield return` for lazy evaluation
- Takes `NeuralNetworkArchitecture<T>` as first parameter
- Industry-standard defaults for each model type:
  - `CreateDefaultLayers` / `CreateDefaultFeedForwardLayers` - Feed-forward networks
  - `CreateDefaultCNNLayers` - Convolutional Neural Networks
  - `CreateDefaultResNetLayers` - Residual Networks
  - `CreateDefaultOccupancyTemporalLayers` - LSTM-based temporal networks
  - `CreateDefaultDeepBoltzmannMachineLayers` - DBMs
  - `CreateDefaultAttentionLayers` - Transformer-style networks
- ValidateLayerParameters helper for input validation

### Reference Base Class: NoisePredictorBase

Located at: `src/Diffusion/NoisePredictors/NoisePredictorBase.cs`

Key patterns demonstrated:
- IFullModel implementation (standalone pattern)
- INoisePredictor interface for diffusion models
- Timestep embedding computation
- Numerical gradient computation
- Stream-based serialization
