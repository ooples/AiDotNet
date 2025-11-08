# Junior Developer Implementation Guide: Issue #264
## ControlNet and T2I-Adapter Support for Conditioning

### Overview
ControlNet adds spatial control to text-to-image generation by allowing structural inputs like edges, depth maps, poses, or scribbles to guide the generation process.

---

## Understanding ControlNet

### The Core Problem

**Text-only control is limited:**
- "A person standing" - where? what pose? what angle?
- Hard to get precise compositions
- Difficult to match reference images

**ControlNet Solution:**
- Provide an edge map and model generates image matching those edges
- Provide a depth map and model respects spatial layout
- Provide a pose skeleton and model places person in that exact pose

**Analogy:** Text-only = "Draw me a cat" (vague). ControlNet = "Draw me a cat following this outline" (precise).

---

## Architecture Overview

```
Input Image → [Condition Processor] → Condition Map
                                              ↓
Text Prompt → [CLIP Encoder] → Text Embeddings
                                              ↓
Random Noise → [U-Net + ControlNet] → Image
                        ↑
                   [Zero Convolutions]
```

### Key Innovation: Zero Convolutions

**Problem:** Adding new parameters can destroy pre-trained model

**Solution:** Initialize new layers with zero weights
- Initially, ControlNet adds nothing (identity function)
- During training, gradually learns useful signals
- Preserves pre-trained quality while adding control

---

## Implementation Details

### Condition Types

1. **Canny Edges** - Preserve structure and outlines
2. **Depth Maps** - Control spatial layout and perspective
3. **Pose Skeletons** - Control character poses
4. **Semantic Segmentation** - Control layout and object placement
5. **Scribbles** - Quick sketches to detailed images

### File Structure

```
src/
├── Interfaces/
│   ├── IConditionProcessor.cs
│   └── IControlNet.cs
├── Diffusion/
│   └── Conditioning/
│       ├── CannyEdgeProcessor.cs
│       ├── DepthMapProcessor.cs
│       ├── PoseProcessor.cs
│       └── ScribbleProcessor.cs
├── NeuralNetworks/
│   └── Architectures/
│       └── ControlNet/
│           ├── ControlNetBlock.cs
│           ├── ZeroConvolution.cs
│           └── ControlNetModel.cs
└── Models/
    └── Generative/
        └── Diffusion/
            └── ControlNetDiffusion.cs
```

---

## Key Implementation Steps

### Step 1: Condition Processor Interface

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Processes condition inputs for ControlNet.
/// </summary>
public interface IConditionProcessor<T>
{
    /// <summary>
    /// Processes raw image into condition map.
    /// </summary>
    Tensor<T> Process(Tensor<T> image);

    /// <summary>
    /// Gets the condition type name (e.g., "canny", "depth", "pose").
    /// </summary>
    string ConditionType { get; }
}
```

### Step 2: Canny Edge Processor Example

```csharp
namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// Extracts edges using Canny edge detector.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Canny is a popular edge detection algorithm.
///
/// Steps:
/// 1. Convert to grayscale
/// 2. Apply Gaussian blur (reduce noise)
/// 3. Compute gradients (intensity changes)
/// 4. Non-maximum suppression (thin edges)
/// 5. Hysteresis thresholding (keep strong edges)
///
/// Output: Binary map where 1 = edge, 0 = no edge
/// </remarks>
public class CannyEdgeProcessor<T> : IConditionProcessor<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _lowThreshold;
    private readonly T _highThreshold;

    public string ConditionType => "canny";

    /// <summary>
    /// Initializes Canny edge processor.
    /// </summary>
    /// <param name="lowThreshold">Low threshold for hysteresis. Default: 100.</param>
    /// <param name="highThreshold">High threshold for hysteresis. Default: 200.</param>
    public CannyEdgeProcessor(T lowThreshold = default(T), T highThreshold = default(T))
    {
        _lowThreshold = NumOps.Equals(lowThreshold, default(T))
            ? NumOps.FromDouble(100.0)
            : lowThreshold;
        _highThreshold = NumOps.Equals(highThreshold, default(T))
            ? NumOps.FromDouble(200.0)
            : highThreshold;
    }

    public Tensor<T> Process(Tensor<T> image)
    {
        // 1. Convert to grayscale if color
        var gray = ConvertToGrayscale(image);

        // 2. Apply Gaussian blur (sigma = 1.4)
        var blurred = ApplyGaussianBlur(gray, NumOps.FromDouble(1.4));

        // 3. Compute gradients using Sobel operators
        var (gradX, gradY) = ComputeSobelGradients(blurred);

        // 4. Non-maximum suppression
        var suppressed = NonMaximumSuppression(gradX, gradY);

        // 5. Hysteresis thresholding
        var edges = HysteresisThreshold(suppressed, _lowThreshold, _highThreshold);

        return edges;
    }

    private Tensor<T> ConvertToGrayscale(Tensor<T> image)
    {
        // RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
        // Implementation details omitted
        return image;
    }

    // Additional helper methods...
}
```

### Step 3: Zero Convolution Layer

```csharp
namespace AiDotNet.NeuralNetworks.Architectures.ControlNet;

/// <summary>
/// Convolution layer initialized with zero weights.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Zero convolutions are crucial for ControlNet.
///
/// Why zero initialization?
/// - Preserves pre-trained model quality
/// - ControlNet starts as identity (adds nothing)
/// - Gradually learns useful signals during training
/// - Prevents catastrophic forgetting
///
/// Without this, adding ControlNet would destroy the pre-trained model!
/// </remarks>
public class ZeroConvolution<T> : ConvolutionalLayer<T>
{
    public ZeroConvolution(int inputDepth, int outputDepth, int kernelSize = 1)
        : base(inputDepth, outputDepth, kernelSize, stride: 1, padding: 0)
    {
        InitializeToZero();
    }

    private void InitializeToZero()
    {
        // Set all weights and biases to zero
        var parameters = GetParameters();
        for (int i = 0; i < parameters.Length; i++)
        {
            parameters[i] = NumOps.Zero;
        }
    }
}
```

### Step 4: ControlNet Block

```csharp
namespace AiDotNet.NeuralNetworks.Architectures.ControlNet;

/// <summary>
/// ControlNet conditioning block that mirrors U-Net encoder.
/// </summary>
public class ControlNetBlock<T> : LayerBase<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly ILayer<T>[] _encoderLayers; // Copy of U-Net encoder
    private readonly ZeroConvolution<T> _zeroConv; // Zero-initialized output

    /// <summary>
    /// Creates ControlNet block matching U-Net encoder structure.
    /// </summary>
    /// <param name="channels">Number of input channels.</param>
    /// <param name="outChannels">Number of output channels.</param>
    public ControlNetBlock(int channels, int outChannels)
    {
        // Copy U-Net encoder structure
        _encoderLayers = CreateEncoderCopy(channels, outChannels);

        // Zero convolution for gradual learning
        _zeroConv = new ZeroConvolution<T>(outChannels, outChannels, kernelSize: 1);
    }

    private ILayer<T>[] CreateEncoderCopy(int channels, int outChannels)
    {
        // Create layers matching U-Net encoder
        // Typically: Conv → GroupNorm → SiLU → Conv → GroupNorm
        return new ILayer<T>[]
        {
            new ConvolutionalLayer<T>(channels, outChannels, kernelSize: 3, padding: 1),
            new GroupNormalizationLayer<T>(outChannels, numGroups: 32),
            new ActivationLayer<T>(ActivationFunction.SiLU),
            new ConvolutionalLayer<T>(outChannels, outChannels, kernelSize: 3, padding: 1),
            new GroupNormalizationLayer<T>(outChannels, numGroups: 32)
        };
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        var x = input;

        // Forward through encoder layers
        foreach (var layer in _encoderLayers)
        {
            x = layer.Forward(x);
        }

        // Zero convolution output (starts at zero, learns gradually)
        var output = _zeroConv.Forward(x);

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> input, Tensor<T> gradients)
    {
        // Backprop through zero conv and encoder layers
        throw new NotImplementedException();
    }

    public override int ParameterCount =>
        _encoderLayers.Sum(l => l.ParameterCount) + _zeroConv.ParameterCount;
}
```

### Step 5: ControlNet-Conditioned Diffusion

```csharp
namespace AiDotNet.Models.Generative.Diffusion;

/// <summary>
/// Diffusion model with ControlNet spatial conditioning.
/// </summary>
public class ControlNetDiffusion<T> : TextConditionedDiffusion<T>
{
    private readonly ControlNetBlock<T>[] _controlNetBlocks;
    private readonly Dictionary<string, IConditionProcessor<T>> _conditionProcessors;

    /// <summary>
    /// Initializes ControlNet diffusion model.
    /// </summary>
    public ControlNetDiffusion(
        IStepScheduler<T> scheduler,
        ITextEncoder<T> textEncoder,
        IVAE<T> vae,
        INeuralNetwork<T> unet,
        ControlNetBlock<T>[] controlNetBlocks)
        : base(scheduler, textEncoder, vae, unet)
    {
        _controlNetBlocks = controlNetBlocks ?? throw new ArgumentNullException(nameof(controlNetBlocks));
        _conditionProcessors = new Dictionary<string, IConditionProcessor<T>>();

        // Register default processors
        RegisterConditionProcessor(new CannyEdgeProcessor<T>());
        RegisterConditionProcessor(new DepthMapProcessor<T>());
    }

    public void RegisterConditionProcessor(IConditionProcessor<T> processor)
    {
        _conditionProcessors[processor.ConditionType] = processor;
    }

    /// <summary>
    /// Generates image with spatial conditioning.
    /// </summary>
    /// <param name="prompt">Text prompt.</param>
    /// <param name="conditionImage">Conditioning image (e.g., edge map, depth map).</param>
    /// <param name="conditionType">Type of condition ("canny", "depth", "pose").</param>
    /// <param name="conditionScale">
    /// Strength of conditioning. Default: 1.0.
    /// - 0.0 = ignore condition (text-only)
    /// - 1.0 = balanced (recommended)
    /// - 2.0 = very strong conditioning (less creativity)
    /// </param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">CFG scale for text.</param>
    /// <returns>Generated image.</returns>
    public Tensor<T> Generate(
        string prompt,
        Tensor<T> conditionImage,
        string conditionType,
        T conditionScale = default(T),
        int numInferenceSteps = 50,
        T guidanceScale = default(T))
    {
        if (!_conditionProcessors.ContainsKey(conditionType))
            throw new ArgumentException($"Unknown condition type: {conditionType}. Register processor first.");

        // Default scales
        if (NumOps.Equals(conditionScale, default(T)))
            conditionScale = NumOps.One;
        if (NumOps.Equals(guidanceScale, default(T)))
            guidanceScale = NumOps.FromDouble(7.5);

        // Process condition
        var processor = _conditionProcessors[conditionType];
        var conditionMap = processor.Process(conditionImage);

        // Encode text
        var textEmbeddings = _textEncoder.Encode(prompt);
        var unconditionalEmbeddings = _textEncoder.Encode("");

        // Initialize noise
        var latents = _scheduler.InitNoise(new[] { 1, 4, 64, 64 });

        // Get timesteps
        var timesteps = _scheduler.GetTimesteps(numInferenceSteps);

        // Denoising loop with ControlNet
        for (int i = 0; i < timesteps.Length; i++)
        {
            int t = timesteps[i];

            // Forward through ControlNet
            var controlOutputs = ForwardControlNet(conditionMap, t);

            // Scale control outputs
            for (int j = 0; j < controlOutputs.Length; j++)
            {
                controlOutputs[j] = controlOutputs[j].Multiply(conditionScale);
            }

            // Predict noise with and without text (CFG)
            var noisePredUncond = ForwardUNetWithControl(
                latents, t, unconditionalEmbeddings, controlOutputs);
            var noisePredCond = ForwardUNetWithControl(
                latents, t, textEmbeddings, controlOutputs);

            // Apply classifier-free guidance
            var noisePred = ApplyClassifierFreeGuidance(
                noisePredUncond, noisePredCond, guidanceScale);

            // Denoise step
            latents = _scheduler.Step(noisePred, t, latents);
        }

        return _vae.Decode(latents);
    }

    private Tensor<T>[] ForwardControlNet(Tensor<T> condition, int timestep)
    {
        // Pass condition through all ControlNet blocks
        var outputs = new List<Tensor<T>>();
        var x = condition;

        foreach (var block in _controlNetBlocks)
        {
            x = block.Forward(x);
            outputs.Add(x);
        }

        return outputs.ToArray();
    }

    private Tensor<T> ForwardUNetWithControl(
        Tensor<T> latents,
        int timestep,
        Tensor<T> textEmbeddings,
        Tensor<T>[] controlOutputs)
    {
        // U-Net forward pass with ControlNet additions
        // At each encoder level, ADD control output to U-Net features

        // Pseudo-code:
        // for (int level = 0; level < encoderLevels; level++)
        // {
        //     unetFeatures = unetEncoder[level].Forward(x);
        //     unetFeatures = unetFeatures.Add(controlOutputs[level]);
        //     x = unetFeatures;
        // }

        // Placeholder - full implementation requires U-Net refactoring
        return latents;
    }
}
```

---

## Testing Strategy

```csharp
[Fact]
public void CannyProcessor_ExtractsEdges()
{
    var processor = new CannyEdgeProcessor<double>();
    var image = CreateTestImage(height: 64, width: 64, channels: 3);

    var edges = processor.Process(image);

    Assert.Equal(64, edges.Shape[0]); // Height
    Assert.Equal(64, edges.Shape[1]); // Width
}

[Fact]
public void ZeroConvolution_InitiallyZero()
{
    var zeroConv = new ZeroConvolution<double>(
        inputDepth: 64,
        outputDepth: 64,
        kernelSize: 1
    );

    var input = new Tensor<double>(new[] { 1, 64, 32, 32 });
    var output = zeroConv.Forward(input);

    // All outputs should be zero initially
    Assert.All(output.ToArray(), x => Assert.Equal(0.0, x, precision: 10));
}

[Fact]
public void ControlNetBlock_PreservesShape()
{
    var block = new ControlNetBlock<double>(channels: 64, outChannels: 128);
    var input = new Tensor<double>(new[] { 1, 64, 32, 32 });

    var output = block.Forward(input);

    Assert.Equal(1, output.Shape[0]); // Batch
    Assert.Equal(128, output.Shape[1]); // Channels
    Assert.Equal(32, output.Shape[2]); // Height
    Assert.Equal(32, output.Shape[3]); // Width
}
```

---

## Training ControlNet

### Two-Phase Training

**Phase 1: Freeze Base Model**
```csharp
// Freeze U-Net and VAE
unet.Freeze();
vae.Freeze();

// Train only ControlNet
foreach (var block in controlNetBlocks)
{
    block.Unfreeze();
}

// Training loop
for (int epoch = 0; epoch < numEpochs; epoch++)
{
    foreach (var batch in trainingData)
    {
        // Extract condition from clean image
        var condition = processor.Process(batch.Image);

        // Add noise to image
        var noisyLatent = scheduler.AddNoise(batch.Latent, noise, timestep);

        // Forward with ControlNet
        var controlOutputs = ForwardControlNet(condition, timestep);
        var noisePred = ForwardUNetWithControl(noisyLatent, timestep, textEmb, controlOutputs);

        // Compute loss
        var loss = MSE(noisePred, noise);

        // Backprop (only ControlNet weights update)
        loss.Backward();
        optimizer.Step();
    }
}
```

**Phase 2: Optional Fine-tuning**
```csharp
// Unfreeze some U-Net layers
unet.UnfreezeTopLayers(numLayers: 2);

// Continue training with lower learning rate
optimizer.SetLearningRate(1e-6); // Much lower
```

---

## Common Pitfalls

### Pitfall 1: Not Using Zero Convolutions
**Wrong:** Regular initialization destroys pre-trained quality
**Correct:** Zero initialization preserves quality

### Pitfall 2: Condition Scale Too High
```csharp
// WRONG: Scale too high, images become rigid
var image = model.Generate(..., conditionScale: 5.0);

// CORRECT: Balanced scale
var image = model.Generate(..., conditionScale: 1.0);
```

### Pitfall 3: Mismatched Resolutions
Condition map must match U-Net encoder resolution at each level (downsampled progressively).

### Pitfall 4: Forgetting to Process Conditions
```csharp
// WRONG: Use raw image directly
var image = model.Generate(..., conditionImage: rawPhoto, ...);

// CORRECT: Process first
var edges = cannyProcessor.Process(rawPhoto);
var image = model.Generate(..., conditionImage: edges, ...);
```

---

## Practical Use Cases

### Use Case 1: Architectural Visualization
```csharp
// Input: Floor plan edges
// Output: Photorealistic rendering
var edges = ExtractEdges(floorPlan);
var rendering = model.Generate(
    prompt: "modern luxury apartment, photorealistic",
    conditionImage: edges,
    conditionType: "canny"
);
```

### Use Case 2: Character Pose Control
```csharp
// Input: Pose skeleton
// Output: Character in exact pose
var pose = ExtractPose(referenceImage);
var character = model.Generate(
    prompt: "superhero in dynamic action pose",
    conditionImage: pose,
    conditionType: "pose"
);
```

### Use Case 3: Depth-Guided Scene Generation
```csharp
// Input: Depth map
// Output: Scene with correct spatial layout
var depth = EstimateDepth(sceneLayout);
var scene = model.Generate(
    prompt: "cozy living room, warm lighting",
    conditionImage: depth,
    conditionType: "depth"
);
```

---

## Next Steps

1. Implement remaining condition processors (depth, pose, segmentation)
2. Add multi-condition support (combine edges + depth)
3. Implement T2I-Adapter (lightweight alternative)
4. Train ControlNet on custom datasets
5. Optimize inference speed

---

## Resources

- **ControlNet Paper**: "Adding Conditional Control to Text-to-Image Diffusion Models" (Zhang et al., 2023)
- **T2I-Adapter**: "T2I-Adapter: Learning Adapters to Dig out More Controllable Ability" (Mou et al., 2023)
- **OpenPose**: For pose extraction
- **MiDaS**: For depth estimation

ControlNet is one of the most impactful innovations in generative AI, enabling precise control over image generation!
