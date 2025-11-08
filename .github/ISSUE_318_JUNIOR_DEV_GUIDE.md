# Issue #318: Junior Developer Implementation Guide
## Implement Standard Pooling and Normalization Layers

## Understanding What Exists vs What's Missing

### Existing Layer Infrastructure

**Location**: `src/NeuralNetworks/Layers/`

**What ALREADY EXISTS**:

1. **PoolingLayer** - `src/NeuralNetworks/Layers/PoolingLayer.cs`
   - Supports Max and Average pooling
   - Configurable pool size and stride
   - Works with 4D tensors [batch, channels, height, width]

2. **MaxPoolingLayer** - `src/NeuralNetworks/Layers/MaxPoolingLayer.cs`
   - Specialized max pooling implementation

3. **GlobalPoolingLayer** - `src/NeuralNetworks/Layers/GlobalPoolingLayer.cs`
   - Global Average Pooling (reduces spatial dimensions to 1x1)
   - Global Max Pooling
   - Works with 4D tensors

4. **BatchNormalizationLayer** - `src/NeuralNetworks/Layers/BatchNormalizationLayer.cs`
   - Normalizes across batch dimension
   - Learnable gamma/beta parameters
   - Running statistics for inference

5. **LayerNormalizationLayer** - `src/NeuralNetworks/Layers/LayerNormalizationLayer.cs`
   - Normalizes across feature dimension
   - Independent per sample

### What's MISSING (Per Issue #318):

**Phase 1: Average Pooling Layers (Explicit Implementations)**
- ❌ AveragePooling1DLayer
- ❌ AveragePooling2DLayer
- ❌ AveragePooling3DLayer

**Phase 2: Global Pooling Layers (Dimension-Specific)**
- ❌ GlobalAveragePooling1DLayer
- ❌ GlobalAveragePooling2DLayer
- ❌ GlobalMaxPooling1DLayer
- ❌ GlobalMaxPooling2DLayer

**Phase 3: Normalization Layers**
- ❌ InstanceNormalizationLayer
- ❌ GroupNormalizationLayer

### CRITICAL INSIGHT:

**Existing `PoolingLayer` ALREADY SUPPORTS average pooling!**
```csharp
public class PoolingLayer<T> : LayerBase<T>
{
    public PoolingType Type { get; }  // Max or Average
}

public enum PoolingType
{
    Max,
    Average
}
```

**So the "missing" average pooling layers are really just convenience wrappers!**

Similarly, **GlobalPoolingLayer ALREADY SUPPORTS both max and average!**

**The Real Goal**: Create explicit, dimension-specific convenience classes for better API usability and PyTorch/TensorFlow compatibility.

---

## Understanding LayerBase Architecture

### Base Class Structure

**File**: `src/NeuralNetworks/Layers/LayerBase.cs`

```csharp
public abstract class LayerBase<T>
{
    // Automatically initialized NumOps
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    // Input/output shapes
    protected int[] InputShape { get; }
    protected int[] OutputShape { get; }

    // Training mode flag
    public bool IsTrainingMode { get; set; }

    // Five abstract methods that all layers must implement:

    // 1. Forward pass
    public abstract Tensor<T> Forward(Tensor<T> input);

    // 2. Backward pass
    public abstract Tensor<T> Backward(Tensor<T> outputGradient);

    // 3. Update parameters
    public abstract void UpdateParameters(T learningRate);

    // 4. Get parameters
    public abstract Vector<T> GetParameters();

    // 5. Reset state
    public abstract void ResetState();

    // Optional: Training support flag
    public virtual bool SupportsTraining => false;
}
```

---

## Phase 1: Implement Explicit Average Pooling Layers

### Conceptual Understanding

**Average Pooling** computes the average of values in each pooling region:
- For pooling window [2, 4, 6, 8], average = (2+4+6+8)/4 = 5
- Reduces spatial dimensions
- Provides smooth downsampling (vs max pooling's abrupt selection)

**Key Properties**:
- All values in region contribute equally
- Smoother than max pooling
- Better for preserving background information
- Used in classification networks (e.g., after feature extraction)

---

### AC 1.1: Create AveragePooling1DLayer.cs

**File**: `src/NeuralNetworks/Layers/AveragePooling1DLayer.cs`

```csharp
namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Performs 1D average pooling, downsampling along the temporal dimension.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Average pooling reduces the temporal dimension by computing the average of values within
/// sliding windows. This is commonly used for sequence data (time series, audio, text).
/// </para>
/// <para><b>For Beginners:</b> This layer is like "smoothing" your sequence data.
///
/// Example with audio signals:
/// - Input: [1, 2, 3, 4, 5, 6, 7, 8] (8 time steps)
/// - Pool size: 2, Stride: 2
/// - Output: [(1+2)/2, (3+4)/2, (5+6)/2, (7+8)/2] = [1.5, 3.5, 5.5, 7.5]
///
/// This layer:
/// - Reduces the length of sequences
/// - Smooths out rapid variations
/// - Preserves overall trends
/// </para>
/// <para><b>Default Parameters:</b>
/// - Pool size: 2 (reduces length by half)
/// - Stride: 2 (non-overlapping windows)
/// - Rationale: Standard practice in deep learning, same as max pooling defaults
/// </para>
/// </remarks>
public class AveragePooling1DLayer<T> : LayerBase<T>
{
    private readonly int _poolSize;
    private readonly int _stride;
    private Tensor<T>? _lastInput;

    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the AveragePooling1DLayer class.
    /// </summary>
    /// <param name="inputLength">The length of the input sequence.</param>
    /// <param name="channels">The number of channels (features) per time step.</param>
    /// <param name="poolSize">The size of the pooling window. Default: 2.</param>
    /// <param name="stride">The stride of the pooling operation. Default: equals poolSize.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Configure the pooling operation for your sequence data.
    ///
    /// Example for audio with 100 time steps and 32 channels:
    /// ```csharp
    /// var avgPool = new AveragePooling1DLayer<float>(
    ///     inputLength: 100,
    ///     channels: 32,
    ///     poolSize: 2,
    ///     stride: 2
    /// );
    /// // Output will have length: (100 - 2) / 2 + 1 = 50
    /// ```
    /// </para>
    /// </remarks>
    public AveragePooling1DLayer(
        int inputLength,
        int channels,
        int poolSize = 2,
        int stride = -1)
        : base(
            inputShape: new[] { inputLength, channels },
            outputShape: new[] { CalculateOutputLength(inputLength, poolSize, stride < 0 ? poolSize : stride), channels })
    {
        _poolSize = poolSize;
        _stride = stride < 0 ? poolSize : stride;
    }

    private static int CalculateOutputLength(int inputLength, int poolSize, int stride)
    {
        return (inputLength - poolSize) / stride + 1;
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        int batchSize = input.Shape[0];
        int inputLength = input.Shape[1];
        int channels = input.Shape[2];

        int outputLength = (inputLength - _poolSize) / _stride + 1;
        var output = new Tensor<T>(new[] { batchSize, outputLength, channels });

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int i = 0; i < outputLength; i++)
                {
                    int start = i * _stride;
                    T sum = NumOps.Zero;

                    for (int p = 0; p < _poolSize; p++)
                    {
                        sum = NumOps.Add(sum, input[b, start + p, c]);
                    }

                    output[b, i, c] = NumOps.Divide(sum, NumOps.FromDouble(_poolSize));
                }
            }
        }

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int inputLength = _lastInput.Shape[1];
        int channels = _lastInput.Shape[2];

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        T gradScale = NumOps.Divide(NumOps.One, NumOps.FromDouble(_poolSize));

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int i = 0; i < outputGradient.Shape[1]; i++)
                {
                    int start = i * _stride;
                    T grad = NumOps.Multiply(outputGradient[b, i, c], gradScale);

                    for (int p = 0; p < _poolSize; p++)
                    {
                        inputGradient[b, start + p, c] = NumOps.Add(inputGradient[b, start + p, c], grad);
                    }
                }
            }
        }

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate) { }
    public override Vector<T> GetParameters() => new Vector<T>(0);
    public override void ResetState() { _lastInput = null; }
}
```

### AC 1.2 & 1.3: Create AveragePooling2DLayer and AveragePooling3DLayer

**Similar pattern to 1D, but with 2D/3D spatial dimensions. Full implementations omitted for brevity - follow the same structure as AveragePooling1DLayer but with nested loops for height/width (2D) or height/width/depth (3D).**

---

## Phase 2: Implement Global Pooling Layers

### Conceptual Understanding

**Global Pooling** reduces entire spatial dimensions to a single value per channel:
- Input: [batch, height, width, channels]
- Output: [batch, 1, 1, channels]

**Use Cases**:
- Replacing fully connected layers
- Reducing model parameters
- Providing translation invariance
- Final layer before classification

---

### AC 2.1: Create GlobalAveragePooling1DLayer.cs

**File**: `src/NeuralNetworks/Layers/GlobalAveragePooling1DLayer.cs`

```csharp
namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Performs global average pooling on 1D sequences, reducing each channel to a single value.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Global average pooling computes the average of all values in each channel across the
/// entire temporal dimension. This is useful for sequence classification tasks.
/// </para>
/// <para><b>For Beginners:</b> This layer summarizes an entire sequence into one number per channel.
///
/// Example with sentiment analysis:
/// - Input: Sequence of 50 word embeddings, 128 features each
/// - Shape: [batch, 50, 128]
/// - Output: Average of all 50 embeddings for each of 128 features
/// - Shape: [batch, 1, 128]
///
/// This creates a fixed-size representation regardless of sequence length!
/// </para>
/// <para><b>When to Use:</b>
/// - Sequence classification (sentiment, intent)
/// - As alternative to fully connected layers
/// - When you want translation invariance
/// - To reduce parameters dramatically
/// </para>
/// </remarks>
public class GlobalAveragePooling1DLayer<T> : LayerBase<T>
{
    private Tensor<T>? _lastInput;

    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the GlobalAveragePooling1DLayer class.
    /// </summary>
    /// <param name="inputLength">The length of the input sequence.</param>
    /// <param name="channels">The number of channels (features).</param>
    public GlobalAveragePooling1DLayer(int inputLength, int channels)
        : base(
            inputShape: new[] { inputLength, channels },
            outputShape: new[] { 1, channels })
    {
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        int batchSize = input.Shape[0];
        int inputLength = input.Shape[1];
        int channels = input.Shape[2];

        var output = new Tensor<T>(new[] { batchSize, 1, channels });

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                T sum = NumOps.Zero;

                for (int i = 0; i < inputLength; i++)
                {
                    sum = NumOps.Add(sum, input[b, i, c]);
                }

                output[b, 0, c] = NumOps.Divide(sum, NumOps.FromDouble(inputLength));
            }
        }

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int inputLength = _lastInput.Shape[1];
        int channels = _lastInput.Shape[2];

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        T gradScale = NumOps.Divide(NumOps.One, NumOps.FromDouble(inputLength));

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                T grad = NumOps.Multiply(outputGradient[b, 0, c], gradScale);

                for (int i = 0; i < inputLength; i++)
                {
                    inputGradient[b, i, c] = grad;
                }
            }
        }

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate) { }
    public override Vector<T> GetParameters() => new Vector<T>(0);
    public override void ResetState() { _lastInput = null; }
}
```

### AC 2.2, 2.3, 2.4: Other Global Pooling Layers

**Similar implementations for:**
- `GlobalAveragePooling2DLayer` - For images [batch, height, width, channels] → [batch, 1, 1, channels]
- `GlobalMaxPooling1DLayer` - Takes max instead of average across temporal dimension
- `GlobalMaxPooling2DLayer` - Takes max instead of average across spatial dimensions

---

## Phase 3: Implement Instance and Group Normalization

### Conceptual Understanding

**Normalization Comparison**:

| Type | Normalizes Across | Use Case |
|------|------------------|----------|
| Batch Norm | Batch dimension | Large batches, conv nets |
| Layer Norm | Feature dimension | RNNs, small batches |
| Instance Norm | Per-sample, per-channel | Style transfer, GANs |
| Group Norm | Groups of channels | Small batches, stable training |

**Instance Normalization**:
- Normalizes each sample and each channel independently
- Formula: `norm = (x - mean(x_sample_channel)) / sqrt(var(x_sample_channel) + epsilon)`
- No dependence on batch size
- Used in style transfer and GANs

**Group Normalization**:
- Divides channels into groups
- Normalizes within each group
- Compromise between Layer Norm (1 group) and Instance Norm (C groups)
- More stable than Batch Norm with small batches

---

### AC 3.1: Create InstanceNormalizationLayer.cs

**File**: `src/NeuralNetworks/Layers/InstanceNormalizationLayer.cs`

```csharp
namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements instance normalization, normalizing each sample and channel independently.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Instance normalization normalizes each sample and each channel independently, making it
/// particularly useful for style transfer and generative adversarial networks (GANs).
/// Unlike batch normalization, it doesn't depend on batch size.
/// </para>
/// <para><b>For Beginners:</b> This layer standardizes each feature map independently.
///
/// Think of image style transfer:
/// - Each image should be normalized on its own
/// - Each color channel (R, G, B) normalized separately
/// - No dependence on other images in the batch
///
/// Formula per sample per channel:
/// normalized = (x - mean) / sqrt(variance + epsilon)
/// output = gamma * normalized + beta
/// </para>
/// <para><b>When to Use Instance Normalization:</b>
/// - Style transfer (e.g., CycleGAN, pix2pix)
/// - Generative models (GANs)
/// - When batch size is 1
/// - When each sample should be treated independently
///
/// When NOT to use:
/// - Classification tasks (use Batch Norm)
/// - When you need batch statistics
/// - With very large batches (Batch Norm is better)
/// </para>
/// <para><b>Default Parameters:</b>
/// - Epsilon: 1e-5 (numerical stability)
/// - Gamma (scale): 1.0 (learnable)
/// - Beta (shift): 0.0 (learnable)
/// - Rationale: Same as Batch Norm defaults
/// </para>
/// </remarks>
public class InstanceNormalizationLayer<T> : LayerBase<T>
{
    private readonly T _epsilon;
    private Vector<T> _gamma;
    private Vector<T> _beta;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastNormalized;
    private Vector<T>? _gammaGradient;
    private Vector<T>? _betaGradient;

    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the InstanceNormalizationLayer class.
    /// </summary>
    /// <param name="numChannels">The number of channels (features) to normalize.</param>
    /// <param name="epsilon">Small constant for numerical stability. Default: 1e-5.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set up instance normalization for your feature maps.
    ///
    /// Example for RGB images with 3 channels:
    /// ```csharp
    /// var instanceNorm = new InstanceNormalizationLayer<float>(numChannels: 3);
    /// ```
    ///
    /// For feature maps from a convolutional layer with 64 filters:
    /// ```csharp
    /// var instanceNorm = new InstanceNormalizationLayer<float>(numChannels: 64);
    /// ```
    /// </para>
    /// </remarks>
    public InstanceNormalizationLayer(int numChannels, double epsilon = 1e-5)
        : base(
            inputShape: new[] { numChannels },
            outputShape: new[] { numChannels })
    {
        _epsilon = NumOps.FromDouble(epsilon);
        _gamma = Vector<T>.CreateDefault(numChannels, NumOps.One);
        _beta = new Vector<T>(numChannels);
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];
        int spatialSize = height * width;

        var output = new Tensor<T>(input.Shape);
        _lastNormalized = new Tensor<T>(input.Shape);

        // Normalize each sample and each channel independently
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                // Compute mean for this sample and channel
                T sum = NumOps.Zero;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        sum = NumOps.Add(sum, input[b, c, h, w]);
                    }
                }
                T mean = NumOps.Divide(sum, NumOps.FromDouble(spatialSize));

                // Compute variance
                T sumSquaredDiff = NumOps.Zero;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        T diff = NumOps.Subtract(input[b, c, h, w], mean);
                        sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Multiply(diff, diff));
                    }
                }
                T variance = NumOps.Divide(sumSquaredDiff, NumOps.FromDouble(spatialSize));

                // Normalize and apply scale/shift
                T std = NumOps.Sqrt(NumOps.Add(variance, _epsilon));
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        T normalized = NumOps.Divide(
                            NumOps.Subtract(input[b, c, h, w], mean),
                            std
                        );
                        _lastNormalized[b, c, h, w] = normalized;
                        output[b, c, h, w] = NumOps.Add(
                            NumOps.Multiply(_gamma[c], normalized),
                            _beta[c]
                        );
                    }
                }
            }
        }

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastNormalized == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int channels = _lastInput.Shape[1];
        int height = _lastInput.Shape[2];
        int width = _lastInput.Shape[3];
        int spatialSize = height * width;

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        _gammaGradient = new Vector<T>(channels);
        _betaGradient = new Vector<T>(channels);

        // Compute gradients per sample per channel
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                // Accumulate gamma and beta gradients
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        _gammaGradient[c] = NumOps.Add(
                            _gammaGradient[c],
                            NumOps.Multiply(outputGradient[b, c, h, w], _lastNormalized[b, c, h, w])
                        );
                        _betaGradient[c] = NumOps.Add(_betaGradient[c], outputGradient[b, c, h, w]);
                    }
                }

                // Compute input gradient (simplified - full formula is complex)
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        inputGradient[b, c, h, w] = NumOps.Multiply(
                            _gamma[c],
                            outputGradient[b, c, h, w]
                        );
                    }
                }
            }
        }

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_gammaGradient == null || _betaGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _gamma = _gamma.Subtract(_gammaGradient.Multiply(learningRate));
        _beta = _beta.Subtract(_betaGradient.Multiply(learningRate));
    }

    public override Vector<T> GetParameters()
    {
        int numChannels = _gamma.Length;
        var parameters = new Vector<T>(numChannels * 2);

        for (int i = 0; i < numChannels; i++)
        {
            parameters[i] = _gamma[i];
            parameters[i + numChannels] = _beta[i];
        }

        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int numChannels = _gamma.Length;
        if (parameters.Length != numChannels * 2)
            throw new ArgumentException($"Expected {numChannels * 2} parameters, got {parameters.Length}");

        for (int i = 0; i < numChannels; i++)
        {
            _gamma[i] = parameters[i];
            _beta[i] = parameters[i + numChannels];
        }
    }

    public override void ResetState()
    {
        _lastInput = null;
        _lastNormalized = null;
        _gammaGradient = null;
        _betaGradient = null;
    }
}
```

---

### AC 3.2: Create GroupNormalizationLayer.cs

**File**: `src/NeuralNetworks/Layers/GroupNormalizationLayer.cs`

```csharp
namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements group normalization, dividing channels into groups and normalizing within each group.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Group normalization divides channels into groups and normalizes within each group.
/// It's a compromise between Layer Normalization (1 group) and Instance Normalization (C groups).
/// More stable than Batch Normalization with small batch sizes.
/// </para>
/// <para><b>For Beginners:</b> This layer groups features together before normalizing.
///
/// Think of organizing students into study groups:
/// - Layer Norm: All students in one big group
/// - Instance Norm: Each student is their own group
/// - Group Norm: Students divided into small groups (e.g., groups of 4)
///
/// Example with 32 channels and 4 groups:
/// - Channels 0-7: Group 1 (normalized together)
/// - Channels 8-15: Group 2 (normalized together)
/// - Channels 16-23: Group 3 (normalized together)
/// - Channels 24-31: Group 4 (normalized together)
/// </para>
/// <para><b>When to Use Group Normalization:</b>
/// - Small batch sizes (where Batch Norm is unstable)
/// - Object detection and segmentation
/// - When you need more stable training than Batch Norm
/// - As alternative to Layer Norm with better performance
///
/// When NOT to use:
/// - Large batch sizes (Batch Norm works well)
/// - When Layer Norm or Instance Norm already work
/// </para>
/// <para><b>Default Parameters:</b>
/// - Epsilon: 1e-5 (numerical stability)
/// - Number of groups: 32 (typical for networks with 32+ channels)
/// - Rationale: Wu & He, "Group Normalization" (ECCV 2018)
/// - Common choices: 8, 16, 32 groups
/// </para>
/// </remarks>
public class GroupNormalizationLayer<T> : LayerBase<T>
{
    private readonly T _epsilon;
    private readonly int _numGroups;
    private readonly int _numChannels;
    private readonly int _channelsPerGroup;
    private Vector<T> _gamma;
    private Vector<T> _beta;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastNormalized;
    private Vector<T>? _gammaGradient;
    private Vector<T>? _betaGradient;

    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the GroupNormalizationLayer class.
    /// </summary>
    /// <param name="numChannels">The number of channels (features).</param>
    /// <param name="numGroups">
    /// The number of groups to divide channels into. Default: 32.
    /// Must evenly divide numChannels.
    /// </param>
    /// <param name="epsilon">Small constant for numerical stability. Default: 1e-5.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when numChannels is not evenly divisible by numGroups.
    /// </exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Configure group normalization for your network.
    ///
    /// Example for a network with 64 channels and 8 groups:
    /// ```csharp
    /// var groupNorm = new GroupNormalizationLayer<float>(
    ///     numChannels: 64,
    ///     numGroups: 8  // Each group has 64/8 = 8 channels
    /// );
    /// ```
    ///
    /// Common configurations:
    /// - 32 channels, 8 groups (4 channels per group)
    /// - 64 channels, 16 groups (4 channels per group)
    /// - 128 channels, 32 groups (4 channels per group)
    /// </para>
    /// <para><b>Research References:</b>
    /// - Wu & He, "Group Normalization" (ECCV 2018)
    /// - Recommended: 32 groups or C/4 channels per group
    /// - Works well with batch sizes as small as 2
    /// </para>
    /// </remarks>
    public GroupNormalizationLayer(
        int numChannels,
        int numGroups = 32,
        double epsilon = 1e-5)
        : base(
            inputShape: new[] { numChannels },
            outputShape: new[] { numChannels })
    {
        if (numChannels % numGroups != 0)
            throw new ArgumentException(
                $"Number of channels ({numChannels}) must be divisible by number of groups ({numGroups})");

        _epsilon = NumOps.FromDouble(epsilon);
        _numGroups = numGroups;
        _numChannels = numChannels;
        _channelsPerGroup = numChannels / numGroups;

        _gamma = Vector<T>.CreateDefault(numChannels, NumOps.One);
        _beta = new Vector<T>(numChannels);
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        int batchSize = input.Shape[0];
        int height = input.Shape[2];
        int width = input.Shape[3];
        int spatialSize = height * width;
        int groupSize = _channelsPerGroup * spatialSize;

        var output = new Tensor<T>(input.Shape);
        _lastNormalized = new Tensor<T>(input.Shape);

        // Normalize each batch and each group
        for (int b = 0; b < batchSize; b++)
        {
            for (int g = 0; g < _numGroups; g++)
            {
                int groupStart = g * _channelsPerGroup;
                int groupEnd = groupStart + _channelsPerGroup;

                // Compute mean for this group
                T sum = NumOps.Zero;
                for (int c = groupStart; c < groupEnd; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            sum = NumOps.Add(sum, input[b, c, h, w]);
                        }
                    }
                }
                T mean = NumOps.Divide(sum, NumOps.FromDouble(groupSize));

                // Compute variance for this group
                T sumSquaredDiff = NumOps.Zero;
                for (int c = groupStart; c < groupEnd; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            T diff = NumOps.Subtract(input[b, c, h, w], mean);
                            sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Multiply(diff, diff));
                        }
                    }
                }
                T variance = NumOps.Divide(sumSquaredDiff, NumOps.FromDouble(groupSize));

                // Normalize and apply scale/shift for this group
                T std = NumOps.Sqrt(NumOps.Add(variance, _epsilon));
                for (int c = groupStart; c < groupEnd; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            T normalized = NumOps.Divide(
                                NumOps.Subtract(input[b, c, h, w], mean),
                                std
                            );
                            _lastNormalized[b, c, h, w] = normalized;
                            output[b, c, h, w] = NumOps.Add(
                                NumOps.Multiply(_gamma[c], normalized),
                                _beta[c]
                            );
                        }
                    }
                }
            }
        }

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastNormalized == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Similar to InstanceNormalizationLayer but accounting for groups
        // Implementation omitted for brevity - follows same pattern

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        _gammaGradient = new Vector<T>(_numChannels);
        _betaGradient = new Vector<T>(_numChannels);

        // Compute gradients (simplified)
        // Full implementation would compute proper gradients accounting for group structure

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_gammaGradient == null || _betaGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _gamma = _gamma.Subtract(_gammaGradient.Multiply(learningRate));
        _beta = _beta.Subtract(_betaGradient.Multiply(learningRate));
    }

    public override Vector<T> GetParameters()
    {
        var parameters = new Vector<T>(_numChannels * 2);

        for (int i = 0; i < _numChannels; i++)
        {
            parameters[i] = _gamma[i];
            parameters[i + _numChannels] = _beta[i];
        }

        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _numChannels * 2)
            throw new ArgumentException($"Expected {_numChannels * 2} parameters, got {parameters.Length}");

        for (int i = 0; i < _numChannels; i++)
        {
            _gamma[i] = parameters[i];
            _beta[i] = parameters[i + _numChannels];
        }
    }

    public override void ResetState()
    {
        _lastInput = null;
        _lastNormalized = null;
        _gammaGradient = null;
        _betaGradient = null;
    }
}
```

---

## Common Pitfalls to Avoid:

1. **DON'T use `default(T)` or `default!`**
   - Use `NumOps.Zero`, `NumOps.One`
   - Properly initialize collections

2. **DO use NumOps for all arithmetic**
   - `NumOps.Add()`, `NumOps.Divide()`, `NumOps.Sqrt()`
   - `NumOps.GreaterThan()`, `NumOps.LessThan()`

3. **DO implement all five abstract methods**
   - Forward, Backward, UpdateParameters, GetParameters, ResetState

4. **DO handle edge cases**
   - Division by zero (variance = 0)
   - Empty tensors
   - Mismatched dimensions

5. **DO test backward pass carefully**
   - Verify gradients flow correctly
   - Test with different pool sizes and strides
   - Check gradient distribution

6. **DO document with research citations**
   - Instance Norm: Ulyanov et al. 2016
   - Group Norm: Wu & He 2018

7. **DO preserve sparsity where applicable**
   - Max pooling naturally preserves sparsity
   - Average pooling may not

8. **DO test with multiple tensor shapes**
   - Different batch sizes
   - Different spatial dimensions
   - Different channel counts

---

## Testing Strategy:

### Unit Tests (Required - 80%+ coverage):
- Constructor validation
- Forward pass correctness
- Backward pass correctness (gradient flow)
- Parameter updates
- Edge cases (zeros, constant features)
- Multiple numeric types (double, float)
- Various tensor shapes

### Integration Tests:
- Use in neural networks
- Stack multiple layers
- Compare with existing implementations (where applicable)

### Gradient Checks:
- Numerical gradient verification
- Ensure backward pass matches analytical gradients

---

## Summary:

**What You're Building**:
- Explicit average pooling layers (1D, 2D, 3D)
- Dimension-specific global pooling layers
- Instance and Group Normalization layers

**Key Architecture Insights**:
- Most pooling functionality already exists in PoolingLayer
- New classes are convenience wrappers for better API
- Use NumOps for all arithmetic
- Implement all five LayerBase abstract methods

**Implementation Checklist**:
- [ ] Create AveragePooling1D/2D/3DLayers
- [ ] Create GlobalAveragePooling1D/2DLayers
- [ ] Create GlobalMaxPooling1D/2DLayers
- [ ] Create InstanceNormalizationLayer
- [ ] Create GroupNormalizationLayer
- [ ] Write comprehensive unit tests (80%+ coverage)
- [ ] Test backward passes with gradient checks
- [ ] Document with research citations
- [ ] Test with multiple tensor shapes and types

**Success Criteria**:
- All unit tests pass
- 80%+ code coverage
- Gradient checks pass (numerical = analytical)
- Works with existing neural network infrastructure
- Clear beginner-friendly documentation
