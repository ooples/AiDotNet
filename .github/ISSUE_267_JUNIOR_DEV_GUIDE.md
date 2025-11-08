# Issue #267: Junior Developer Implementation Guide

## Video Generation via Spatiotemporal U-Net (Text/Image Conditioned)

**This issue creates NEW functionality for generating video sequences from text prompts using diffusion models.**

### What You're Building:
1. **VideoGenerationModel**: A wrapper that orchestrates frame-by-frame video generation using a pre-trained spatiotemporal diffusion model
2. **Text-to-Video Generation**: Convert text descriptions into coherent video sequences
3. **Denoising Loop**: Iteratively refine random noise into video frames using the diffusion process

---

## Understanding Video Generation Concepts

### What is Video as a Sequence of Frames?

**Video** is fundamentally a sequence of images (frames) displayed rapidly to create the illusion of motion.

**Key Concepts**:
- **Frame**: A single image in the video sequence
- **Frame Rate**: How many frames per second (fps) - typically 24-30 fps for smooth video
- **Temporal Dimension**: The time axis that connects frames in sequence
- **Spatial Dimensions**: The height and width of each frame

**Example**:
```
Frame 0: [Image of empty road at t=0.0s]
Frame 1: [Image of car appearing at t=0.033s]
Frame 2: [Image of car moving right at t=0.066s]
Frame 3: [Image of car further right at t=0.100s]
...
```

**Why This Matters**:
- Each frame is a 3D tensor: `[height, width, channels]` (e.g., `[256, 256, 3]` for RGB)
- A video is a 4D tensor: `[num_frames, height, width, channels]`
- Video models must understand both spatial content (what's in each frame) AND temporal relationships (how frames connect over time)

### What is Temporal Coherence?

**Temporal Coherence** means that adjacent frames should be smoothly connected and consistent with each other.

**Good Temporal Coherence**:
```
Frame 1: Person walking, left foot forward
Frame 2: Person walking, right foot forward (natural progression)
Frame 3: Person walking, left foot forward again
```

**Poor Temporal Coherence**:
```
Frame 1: Person walking in park
Frame 2: Suddenly teleporting to beach (no transition)
Frame 3: Different person entirely (inconsistent identity)
```

**Why This Matters**:
- Without temporal coherence, videos look jittery, objects flicker, and motion is unnatural
- Spatiotemporal models learn to maintain consistency between frames
- The diffusion process must denoise the entire video sequence jointly, not frame-by-frame independently

### What is a Spatiotemporal U-Net?

**Spatiotemporal U-Net** extends the traditional 2D U-Net architecture to handle video data by processing both spatial (x, y) and temporal (t) dimensions simultaneously.

**Traditional 2D Convolution** (for images):
```
Kernel shape: [3, 3, input_channels, output_channels]
Slides across: height × width
Captures: Spatial patterns (edges, textures, shapes)
```

**3D Spatiotemporal Convolution** (for video):
```
Kernel shape: [3, 3, 3, input_channels, output_channels]
               ↑  ↑  ↑
            time  h  w
Slides across: time × height × width
Captures: Spatiotemporal patterns (motion, temporal evolution)
```

**How It Works**:
1. **Encoder (Downsampling)**: Compress spatial and temporal dimensions while extracting features
2. **Bottleneck**: Learn compact spatiotemporal representations
3. **Decoder (Upsampling)**: Expand back to full resolution while maintaining temporal coherence
4. **Skip Connections**: Preserve fine-grained spatiotemporal details from encoder to decoder

**Example - Processing 16 frames of 64×64 video**:
```
Input:      [16, 64, 64, 3]  (16 frames, 64×64 RGB)
↓ 3D Conv + Pooling
Level 1:    [16, 32, 32, 64] (same temporal length, smaller spatial)
↓ 3D Conv + Pooling
Level 2:    [8, 16, 16, 128] (compressed both time and space)
↓ 3D Conv + Pooling
Bottleneck: [4, 8, 8, 256]   (highly compressed representation)
↑ 3D Deconv + Skip Connection
Level 2:    [8, 16, 16, 128]
↑ 3D Deconv + Skip Connection
Level 1:    [16, 32, 32, 64]
↑ 3D Deconv + Skip Connection
Output:     [16, 64, 64, 3]  (reconstructed video)
```

**Why Spatiotemporal?**:
- Regular 2D convolutions treat each frame independently (no temporal awareness)
- Spatiotemporal (3D) convolutions see across time, learning motion patterns
- This is crucial for video generation where frames must flow naturally

### What is Diffusion-Based Video Generation?

**Diffusion Models** generate data by learning to reverse a noise-adding process.

**The Process**:

1. **Forward Diffusion (Training Time Only)**:
   - Start with real video: `v_0` (clean)
   - Gradually add noise over T steps: `v_1`, `v_2`, ..., `v_T` (increasingly noisy)
   - Final result: `v_T` (pure noise)

2. **Reverse Diffusion (Inference - What We Implement)**:
   - Start with random noise: `v_T`
   - Model predicts noise at each step
   - Subtract predicted noise to denoise: `v_T` → `v_{T-1}` → ... → `v_1` → `v_0` (clean video)

**Step-by-Step Example (Simplified)**:
```
Step 50 (most noisy):  v_50 = [random static/snow]
                       ↓ Model predicts noise pattern in v_50
Step 49:               v_49 = v_50 - predicted_noise (slightly clearer)
                       ↓ Model predicts noise pattern in v_49
Step 48:               v_48 = v_49 - predicted_noise (even clearer)
...
Step 1:                v_1 = v_2 - predicted_noise (almost clean)
                       ↓ Final denoising step
Step 0 (clean):        v_0 = [coherent video of "car driving down road"]
```

**Text Conditioning**:
- Text prompt: "A car driving down a road"
- Text encoder (CLIP) converts this to embedding: `[1, 768]` vector
- Embedding is fed to diffusion model at each denoising step
- Model learns: "Remove noise to match the text description"

**Why This Works**:
- Model learns what noise patterns correspond to video features
- Text conditioning guides the denoising process toward desired content
- Iterative refinement produces high-quality, coherent videos

### What is the Denoising Scheduler?

The **Scheduler** controls how much noise to remove at each step of the reverse diffusion process.

**Key Concepts**:
- **Timestep**: Current position in the denoising process (t = 50, 49, ..., 1, 0)
- **Noise Schedule**: How noise level changes over time (typically decreases non-linearly)
- **Denoising Strength**: How aggressively to remove noise at each step

**Common Schedulers**:

1. **DDPM (Denoising Diffusion Probabilistic Model)**:
   - Original diffusion scheduler
   - Fixed variance schedule
   - Slower but very stable

2. **DDIM (Denoising Diffusion Implicit Model)**:
   - Deterministic variant
   - Can skip steps (faster inference)
   - Same quality with fewer steps

3. **Linear Schedule** (Simplified):
```
noise_level[t] = (T - t) / T

t=50: noise = 50/50 = 1.0 (100% noise)
t=25: noise = 25/50 = 0.5 (50% noise)
t=0:  noise = 0/50  = 0.0 (0% noise, clean)
```

**Scheduler's Role in Our Implementation**:
```csharp
for (int t = numSteps - 1; t >= 0; t--)
{
    // Get current noise level from scheduler
    float alpha = scheduler.GetAlpha(t);
    float beta = scheduler.GetBeta(t);

    // Predict noise in current video tensor
    var predictedNoise = model.Forward(noisyVideo, t, textEmbedding);

    // Remove noise according to scheduler
    noisyVideo = scheduler.Step(noisyVideo, predictedNoise, t);
}
```

---

## Phase 1: Step-by-Step Implementation

### Understanding the Architecture

**Before coding**, understand how components fit together:

```
Text Prompt: "A car driving down a road"
      ↓
[Text Encoder] → Text Embedding: [1, 768]
      ↓
[Initialize Random Noise] → Noisy Video: [num_frames, height, width, channels]
      ↓
[Denoising Loop] (50 steps):
    For t = 49 to 0:
        ↓
    [Spatiotemporal U-Net](noisyVideo, t, textEmbedding)
        → PredictedNoise: [num_frames, height, width, channels]
        ↓
    [Scheduler.Step](noisyVideo, predictedNoise, t)
        → noisyVideo (slightly denoised)
        ↓
[Output Clean Video] → Frames: [num_frames, height, width, channels]
```

### AC 1.1: Scaffolding the VideoGenerationModel Wrapper

**File**: `src/Models/Video/VideoGenerationModel.cs` (NEW FILE - create `src/Models/Video` directory)

**Step 1: Create the class structure**

```csharp
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;
using System.Collections.Generic;

namespace AiDotNet.Models.Video;

/// <summary>
/// Generates video sequences from text prompts using a pre-trained spatiotemporal diffusion model.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This class wraps a pre-trained video diffusion ONNX model and orchestrates the denoising process
/// to generate coherent video sequences from text descriptions.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a "video artist" that paints moving pictures from your descriptions.
///
/// How it works:
/// 1. You provide a text description: "A sunset over the ocean"
/// 2. The text is converted to a numerical embedding that the model understands
/// 3. The model starts with pure random noise (static)
/// 4. Over 50 steps, it gradually refines the noise into a coherent video
/// 5. Each step removes a bit more noise and adds more detail matching your description
///
/// This is similar to how an artist might:
/// - Start with a rough sketch (noisy)
/// - Gradually add details (denoising)
/// - Keep the description in mind throughout (text conditioning)
/// - Ensure all frames connect smoothly (temporal coherence)
/// </para>
/// </remarks>
public class VideoGenerationModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    // ONNX model for video diffusion (predicts noise at each timestep)
    private readonly OnnxModel<T> _videoDiffusion;

    // ONNX model for encoding text prompts (e.g., CLIP text encoder)
    private readonly OnnxModel<T> _textEncoder;

    // Default values for video generation
    private readonly int _defaultHeight;
    private readonly int _defaultWidth;
    private readonly int _defaultChannels;

    /// <summary>
    /// Initializes a new instance of the VideoGenerationModel class.
    /// </summary>
    /// <param name="videoDiffusionOnnxPath">Path to the pre-trained video diffusion ONNX model.</param>
    /// <param name="textEncoderPath">Path to the text encoder ONNX model (e.g., CLIP).</param>
    /// <param name="defaultHeight">Default height of generated video frames (default: 256).</param>
    /// <param name="defaultWidth">Default width of generated video frames (default: 256).</param>
    /// <param name="defaultChannels">Default number of channels (default: 3 for RGB).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the video generator by loading the necessary models.
    ///
    /// You need two models:
    /// 1. Video Diffusion Model: The main model that generates video frames
    /// 2. Text Encoder: Converts your text prompt into numbers the model understands
    ///
    /// The default dimensions (256×256×3) mean:
    /// - 256 pixels tall
    /// - 256 pixels wide
    /// - 3 channels (Red, Green, Blue)
    ///
    /// You can adjust these based on what the pre-trained model was designed for.
    /// </para>
    /// </remarks>
    public VideoGenerationModel(
        string videoDiffusionOnnxPath,
        string textEncoderPath,
        int defaultHeight = 256,
        int defaultWidth = 256,
        int defaultChannels = 3)
    {
        _videoDiffusion = new OnnxModel<T>(videoDiffusionOnnxPath);
        _textEncoder = new OnnxModel<T>(textEncoderPath);
        _defaultHeight = defaultHeight;
        _defaultWidth = defaultWidth;
        _defaultChannels = defaultChannels;
    }
}
```

**Understanding OnnxModel<T>**:
- ONNX (Open Neural Network Exchange) is a standard format for neural network models
- `OnnxModel<T>` is a wrapper that loads and runs ONNX models (defined in issue #280)
- It has a `Forward(inputs)` method that runs inference and returns outputs
- This allows us to use pre-trained models without reimplementing them

### AC 1.2: Implement the Generate Method

**Step 2: Add text encoding**

```csharp
/// <summary>
/// Encodes a text prompt into an embedding tensor.
/// </summary>
/// <param name="textPrompt">The text description to encode.</param>
/// <returns>A tensor containing the text embedding.</returns>
/// <remarks>
/// <para><b>For Beginners:</b> This converts your text description into numbers.
///
/// Example:
/// - Input: "A car driving down a road"
/// - Output: [0.23, -0.45, 0.12, ...] (768 numbers capturing the meaning)
///
/// The text encoder (like CLIP) has learned to represent text as vectors where:
/// - Similar descriptions have similar vectors
/// - The video model has learned to generate videos matching these vectors
/// </para>
/// </remarks>
private Tensor<T> EncodeText(string textPrompt)
{
    // Text encoder expects tokenized text as input
    // For simplicity, we assume the ONNX model handles tokenization internally
    // In a real implementation, you might need to tokenize the text first

    var inputs = new Dictionary<string, Tensor<T>>
    {
        ["text"] = ConvertTextToTensor(textPrompt)
    };

    var outputs = _textEncoder.Forward(inputs);
    return outputs["text_embedding"];
}

/// <summary>
/// Converts a text string to a tensor format expected by the text encoder.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is a placeholder for text-to-tensor conversion.
/// In a real implementation, this would:
/// 1. Tokenize the text (split into words/subwords)
/// 2. Convert tokens to IDs using a vocabulary
/// 3. Pad/truncate to fixed length
/// 4. Return as a tensor of token IDs
/// </para>
/// </remarks>
private Tensor<T> ConvertTextToTensor(string text)
{
    // Placeholder implementation
    // In reality, this would use a tokenizer from the ONNX model or a separate library
    throw new NotImplementedException(
        "Text tokenization must be implemented based on the specific text encoder model being used. " +
        "This typically involves loading a tokenizer (e.g., CLIP tokenizer) and converting text to token IDs.");
}
```

**Step 3: Initialize random noise**

```csharp
/// <summary>
/// Initializes a random noise tensor for video generation.
/// </summary>
/// <param name="numFrames">Number of frames to generate.</param>
/// <param name="height">Height of each frame.</param>
/// <param name="width">Width of each frame.</param>
/// <param name="channels">Number of channels per frame.</param>
/// <returns>A tensor filled with random noise following a normal distribution.</returns>
/// <remarks>
/// <para><b>For Beginners:</b> This creates the starting point for video generation.
///
/// Think of it like starting with pure static on a TV screen:
/// - Each pixel is set to a random value
/// - The values follow a "bell curve" (normal distribution) around zero
/// - This noise is what the diffusion model will gradually refine into a video
///
/// Why random noise?
/// - The model was trained to denoise random noise into videos
/// - Different random seeds produce different videos from the same prompt
/// - This gives the model creative freedom within the prompt's constraints
/// </para>
/// </remarks>
private Tensor<T> InitializeRandomNoise(int numFrames, int height, int width, int channels)
{
    var shape = new[] { numFrames, height, width, channels };
    var noise = new Tensor<T>(shape);

    // Fill with random values from standard normal distribution (mean=0, std=1)
    var random = new Random();
    for (int i = 0; i < noise.Length; i++)
    {
        // Box-Muller transform to generate normal distribution
        double u1 = random.NextDouble();
        double u2 = random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        noise[i] = NumOps.FromDouble(randStdNormal);
    }

    return noise;
}
```

**Understanding Normal Distribution**:
- Most diffusion models expect noise from a **normal (Gaussian) distribution**
- Mean = 0 (centered around zero)
- Standard deviation = 1 (moderate spread)
- This is the same distribution the model saw during training

**Step 4: Implement the denoising loop**

```csharp
/// <summary>
/// Generates a video sequence from a text prompt.
/// </summary>
/// <param name="textPrompt">The text description of the video to generate.</param>
/// <param name="numFrames">Number of frames to generate (default: 16).</param>
/// <param name="steps">Number of denoising steps (default: 50). More steps = better quality but slower.</param>
/// <param name="height">Height of generated frames (default: uses model default).</param>
/// <param name="width">Width of generated frames (default: uses model default).</param>
/// <returns>An enumerable sequence of frame tensors.</returns>
/// <remarks>
/// <para><b>For Beginners:</b> This is the main method that generates a video from your text description.
///
/// Process:
/// 1. Convert text to embedding
/// 2. Start with random noise
/// 3. Gradually denoise over N steps (typically 50)
/// 4. Return individual frames
///
/// Parameters explained:
/// - textPrompt: What you want to see ("A dog playing in a park")
/// - numFrames: How long the video is (16 frames ≈ 0.5 seconds at 30fps)
/// - steps: How many refinement steps (50 is standard, more = slower but better quality)
///
/// Example usage:
/// ```csharp
/// var model = new VideoGenerationModel<float>("model.onnx", "text_encoder.onnx");
/// var frames = model.Generate("A sunset over the ocean", numFrames: 16, steps: 50);
/// foreach (var frame in frames)
/// {
///     // Save or display each frame
/// }
/// ```
/// </para>
/// </remarks>
public IEnumerable<Tensor<T>> Generate(
    string textPrompt,
    int numFrames,
    int steps = 50,
    int? height = null,
    int? width = null)
{
    // Use defaults if not specified
    int frameHeight = height ?? _defaultHeight;
    int frameWidth = width ?? _defaultWidth;

    // Step 1: Encode text prompt
    var textEmbedding = EncodeText(textPrompt);

    // Step 2: Initialize random noise
    var noisyVideo = InitializeRandomNoise(numFrames, frameHeight, frameWidth, _defaultChannels);

    // Step 3: Create scheduler (simplified linear schedule)
    var scheduler = new SimpleDiffusionScheduler<T>(steps);

    // Step 4: Denoising loop (reverse diffusion)
    for (int t = steps - 1; t >= 0; t--)
    {
        // Create timestep tensor
        var timestep = new Tensor<T>(new[] { 1 });
        timestep[0] = NumOps.FromInt(t);

        // Prepare inputs for the model
        var modelInputs = new Dictionary<string, Tensor<T>>
        {
            ["noisy_video"] = noisyVideo,
            ["timestep"] = timestep,
            ["text_embedding"] = textEmbedding
        };

        // Predict noise in current video tensor
        var modelOutputs = _videoDiffusion.Forward(modelInputs);
        var predictedNoise = modelOutputs["predicted_noise"];

        // Remove noise according to scheduler
        noisyVideo = scheduler.Step(noisyVideo, predictedNoise, t);
    }

    // Step 5: Yield individual frames
    for (int i = 0; i < numFrames; i++)
    {
        yield return ExtractFrame(noisyVideo, i);
    }
}

/// <summary>
/// Extracts a single frame from a video tensor.
/// </summary>
/// <param name="video">The video tensor with shape [num_frames, height, width, channels].</param>
/// <param name="frameIndex">The index of the frame to extract.</param>
/// <returns>A tensor representing the extracted frame with shape [height, width, channels].</returns>
private Tensor<T> ExtractFrame(Tensor<T> video, int frameIndex)
{
    int height = video.Shape[1];
    int width = video.Shape[2];
    int channels = video.Shape[3];

    var frame = new Tensor<T>(new[] { height, width, channels });

    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            for (int c = 0; c < channels; c++)
            {
                frame[h, w, c] = video[frameIndex, h, w, c];
            }
        }
    }

    return frame;
}
```

**Understanding the Denoising Loop**:
1. **Iteration Order**: We iterate **backwards** from `t = steps-1` down to `t = 0`
   - This mimics the reverse diffusion process
   - `t = steps-1` is the noisiest
   - `t = 0` is the cleanest

2. **Model Inputs**:
   - `noisy_video`: Current state of the video (starts as pure noise)
   - `timestep`: Tells the model which denoising step we're on
   - `text_embedding`: Guides the model toward the desired content

3. **Model Output**:
   - `predicted_noise`: The model's prediction of what noise is present
   - We subtract this from the noisy video to denoise it

**Step 5: Implement the simplified scheduler**

```csharp
/// <summary>
/// A simplified linear diffusion scheduler for the denoising process.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The scheduler controls how we remove noise at each step.
///
/// Think of it like a recipe for cleaning:
/// - At step 50 (very noisy): Remove large chunks of noise aggressively
/// - At step 25 (medium noise): Remove moderate amounts carefully
/// - At step 1 (almost clean): Remove tiny amounts very gently
///
/// This gradual approach ensures smooth, high-quality results.
/// </para>
/// </remarks>
private class SimpleDiffusionScheduler<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _numSteps;
    private readonly T[] _alphas;
    private readonly T[] _betas;

    public SimpleDiffusionScheduler(int numSteps)
    {
        _numSteps = numSteps;
        _alphas = new T[numSteps];
        _betas = new T[numSteps];

        // Initialize linear schedule
        for (int t = 0; t < numSteps; t++)
        {
            // Beta increases linearly from small to large
            double betaValue = 0.0001 + (0.02 - 0.0001) * t / (numSteps - 1);
            _betas[t] = NumOps.FromDouble(betaValue);

            // Alpha = 1 - beta
            _alphas[t] = NumOps.Subtract(NumOps.One, _betas[t]);
        }
    }

    /// <summary>
    /// Performs a single denoising step.
    /// </summary>
    /// <param name="noisyVideo">The current noisy video tensor.</param>
    /// <param name="predictedNoise">The noise predicted by the model.</param>
    /// <param name="timestep">The current timestep.</param>
    /// <returns>The denoised video tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is where we actually remove the predicted noise.
    ///
    /// Simplified formula:
    /// cleanerVideo = noisyVideo - (beta / sqrt(alpha)) * predictedNoise
    ///
    /// - beta controls how much noise to remove (larger at later steps)
    /// - alpha controls the scaling
    /// - We subtract the scaled predicted noise from the current video
    /// </para>
    /// </remarks>
    public Tensor<T> Step(Tensor<T> noisyVideo, Tensor<T> predictedNoise, int timestep)
    {
        T alpha = _alphas[timestep];
        T beta = _betas[timestep];

        // Simplified denoising formula
        T scale = NumOps.Divide(beta, NumOps.Sqrt(alpha));

        var denoisedVideo = new Tensor<T>(noisyVideo.Shape);
        for (int i = 0; i < noisyVideo.Length; i++)
        {
            T noiseComponent = NumOps.Multiply(scale, predictedNoise[i]);
            denoisedVideo[i] = NumOps.Subtract(noisyVideo[i], noiseComponent);
        }

        return denoisedVideo;
    }
}
```

**Understanding the Scheduler**:
- **Linear Schedule**: Beta values increase linearly from 0.0001 to 0.02
- **Alpha**: Complement of beta (alpha = 1 - beta)
- **Step Function**: Removes noise proportional to beta and scaled by predicted noise

---

## Phase 2: Testing and Validation

### AC 2.1: Unit Tests

**File**: `tests/UnitTests/Models/VideoGenerationModelTests.cs`

```csharp
using Xunit;
using Moq;
using AiDotNet.Models.Video;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Tests.UnitTests.Models;

/// <summary>
/// Unit tests for VideoGenerationModel.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Unit tests verify that our code works correctly in isolation.
///
/// We use "mocks" to simulate the ONNX models without actually loading them:
/// - This makes tests fast (no need to load large model files)
/// - We can control exactly what the mock models return
/// - We can verify that the correct methods were called
/// </para>
/// </remarks>
public class VideoGenerationModelTests
{
    [Fact]
    public void Generate_CallsOnnxModelCorrectNumberOfTimes()
    {
        // Arrange
        var mockVideoDiffusion = new Mock<IOnnxModel<float>>();
        var mockTextEncoder = new Mock<IOnnxModel<float>>();

        int expectedSteps = 10;
        int numFrames = 4;

        // Setup text encoder to return a dummy embedding
        mockTextEncoder
            .Setup(m => m.Forward(It.IsAny<Dictionary<string, Tensor<float>>>()))
            .Returns(new Dictionary<string, Tensor<float>>
            {
                ["text_embedding"] = new Tensor<float>(new[] { 1, 768 })
            });

        // Setup video diffusion to return predicted noise
        mockVideoDiffusion
            .Setup(m => m.Forward(It.IsAny<Dictionary<string, Tensor<float>>>()))
            .Returns(() => new Dictionary<string, Tensor<float>>
            {
                ["predicted_noise"] = new Tensor<float>(new[] { numFrames, 64, 64, 3 })
            });

        var model = new VideoGenerationModel<float>(mockVideoDiffusion.Object, mockTextEncoder.Object);

        // Act
        var frames = model.Generate("test prompt", numFrames, steps: expectedSteps).ToList();

        // Assert
        mockVideoDiffusion.Verify(
            m => m.Forward(It.IsAny<Dictionary<string, Tensor<float>>>()),
            Times.Exactly(expectedSteps),
            "Video diffusion model should be called once per denoising step");

        Assert.Equal(numFrames, frames.Count);
        Assert.All(frames, frame =>
        {
            Assert.Equal(3, frame.Shape.Length);
            Assert.Equal(64, frame.Shape[0]); // height
            Assert.Equal(64, frame.Shape[1]); // width
            Assert.Equal(3, frame.Shape[2]);  // channels
        });
    }

    [Fact]
    public void Generate_PassesCorrectInputShapesToModel()
    {
        // Arrange
        var mockVideoDiffusion = new Mock<IOnnxModel<float>>();
        var mockTextEncoder = new Mock<IOnnxModel<float>>();

        int numFrames = 8;
        int height = 128;
        int width = 128;

        // Capture the inputs passed to the model
        Dictionary<string, Tensor<float>>? capturedInputs = null;

        mockTextEncoder
            .Setup(m => m.Forward(It.IsAny<Dictionary<string, Tensor<float>>>()))
            .Returns(new Dictionary<string, Tensor<float>>
            {
                ["text_embedding"] = new Tensor<float>(new[] { 1, 768 })
            });

        mockVideoDiffusion
            .Setup(m => m.Forward(It.IsAny<Dictionary<string, Tensor<float>>>()))
            .Callback<Dictionary<string, Tensor<float>>>(inputs => capturedInputs = inputs)
            .Returns(() => new Dictionary<string, Tensor<float>>
            {
                ["predicted_noise"] = new Tensor<float>(new[] { numFrames, height, width, 3 })
            });

        var model = new VideoGenerationModel<float>(mockVideoDiffusion.Object, mockTextEncoder.Object);

        // Act
        var frames = model.Generate("test", numFrames, steps: 1, height: height, width: width).ToList();

        // Assert
        Assert.NotNull(capturedInputs);
        Assert.True(capturedInputs.ContainsKey("noisy_video"));
        Assert.True(capturedInputs.ContainsKey("timestep"));
        Assert.True(capturedInputs.ContainsKey("text_embedding"));

        var noisyVideo = capturedInputs["noisy_video"];
        Assert.Equal(4, noisyVideo.Shape.Length);
        Assert.Equal(numFrames, noisyVideo.Shape[0]);
        Assert.Equal(height, noisyVideo.Shape[1]);
        Assert.Equal(width, noisyVideo.Shape[2]);
        Assert.Equal(3, noisyVideo.Shape[3]);
    }
}
```

**Understanding the Tests**:
1. **First Test**: Verifies the denoising loop runs the correct number of times
2. **Second Test**: Checks that inputs have the correct shapes
3. **Mocking**: We simulate the ONNX models to test our logic without real models

### AC 2.2: Integration Test

**File**: `tests/IntegrationTests/Models/VideoGenerationIntegrationTests.cs`

```csharp
using Xunit;
using AiDotNet.Models.Video;
using AiDotNet.Video;

namespace AiDotNet.Tests.IntegrationTests.Models;

/// <summary>
/// Integration tests for VideoGenerationModel with real ONNX models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Integration tests verify that everything works together with real models.
///
/// These tests require:
/// - Actual pre-trained ONNX model files
/// - More time to run (real inference is slow)
/// - More memory (models can be large)
///
/// We skip these tests if the model files aren't available, so they don't fail in CI.
/// </para>
/// </remarks>
public class VideoGenerationIntegrationTests
{
    private const string ModelPath = "test_data/video_diffusion.onnx";
    private const string TextEncoderPath = "test_data/clip_text_encoder.onnx";

    [Fact(Skip = "Requires pre-trained ONNX models")]
    public void Generate_WithRealModel_ProducesVideoFile()
    {
        // Skip if model files don't exist
        if (!File.Exists(ModelPath) || !File.Exists(TextEncoderPath))
        {
            return;
        }

        // Arrange
        var model = new VideoGenerationModel<float>(ModelPath, TextEncoderPath);
        string outputPath = "test_output_video.gif";

        // Act
        var frames = model.Generate(
            textPrompt: "a car driving down a road",
            numFrames: 16,
            steps: 50
        ).ToList();

        // Assert
        Assert.Equal(16, frames.Count);
        Assert.All(frames, frame =>
        {
            Assert.Equal(256, frame.Shape[0]); // height
            Assert.Equal(256, frame.Shape[1]); // width
            Assert.Equal(3, frame.Shape[2]);   // RGB channels
        });

        // Save as video file for manual verification
        VideoHelper.AssembleVideo(frames, outputPath, frameRate: 30.0);

        Assert.True(File.Exists(outputPath), "Output video file should be created");

        // Manual verification required:
        // Open test_output_video.gif and verify:
        // 1. Video shows a car driving
        // 2. Motion is smooth (temporal coherence)
        // 3. No flickering or artifacts
        // 4. Content matches the prompt
    }
}
```

**Manual Verification Steps**:
1. Run the integration test
2. Open the generated `test_output_video.gif`
3. Check for:
   - **Content Match**: Does it show a car driving?
   - **Temporal Coherence**: Does motion look smooth?
   - **Quality**: Are frames clear without excessive noise?
   - **Consistency**: Does the car maintain appearance across frames?

---

## Common Pitfalls and How to Avoid Them

### Pitfall 1: Treating Video as Independent Frames

**Wrong Approach**:
```csharp
// Generating each frame independently
for (int i = 0; i < numFrames; i++)
{
    var frame = GenerateSingleFrame(textPrompt);
    frames.Add(frame);
}
```

**Problem**: Each frame is generated independently, resulting in:
- No temporal coherence
- Flickering and inconsistency
- Objects teleporting between frames

**Correct Approach**:
```csharp
// Generate entire video sequence jointly
var videoTensor = GenerateVideoSequence(textPrompt, numFrames);
```

**Why**: Spatiotemporal models process all frames together, learning temporal relationships.

### Pitfall 2: Incorrect Noise Distribution

**Wrong Approach**:
```csharp
// Uniform random noise [0, 1]
for (int i = 0; i < noise.Length; i++)
{
    noise[i] = NumOps.FromDouble(random.NextDouble());
}
```

**Problem**: Diffusion models expect **normal distribution** (Gaussian), not uniform.

**Correct Approach**:
```csharp
// Normal distribution (mean=0, std=1) using Box-Muller
double u1 = random.NextDouble();
double u2 = random.NextDouble();
double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
noise[i] = NumOps.FromDouble(randStdNormal);
```

### Pitfall 3: Forward Iteration Instead of Reverse

**Wrong Approach**:
```csharp
for (int t = 0; t < steps; t++)  // Forward!
{
    var noise = model.Forward(video, t, embedding);
    video = scheduler.Step(video, noise, t);
}
```

**Problem**: Diffusion denoising must proceed **backwards** from noisy to clean.

**Correct Approach**:
```csharp
for (int t = steps - 1; t >= 0; t--)  // Backward!
{
    var noise = model.Forward(video, t, embedding);
    video = scheduler.Step(video, noise, t);
}
```

### Pitfall 4: Ignoring Tensor Shapes

**Problem**: Mismatched tensor shapes cause crashes or incorrect results.

**Solution**: Always verify shapes:
```csharp
// Expected shapes:
// noisyVideo: [num_frames, height, width, channels]
// textEmbedding: [1, embedding_dim]
// timestep: [1]
// predictedNoise: [num_frames, height, width, channels] (same as noisyVideo)

Assert.Equal(noisyVideo.Shape, predictedNoise.Shape);
```

---

## Testing Checklist

Before submitting your PR:

- [ ] Unit tests pass with mocked ONNX models
- [ ] Unit tests verify correct number of denoising steps
- [ ] Unit tests check input/output tensor shapes
- [ ] Integration test runs with real model (if available)
- [ ] Generated video file can be opened and viewed
- [ ] Manual verification shows temporal coherence
- [ ] Code coverage >= 90%
- [ ] XML documentation for all public methods
- [ ] No hardcoded numeric types (use generic T)
- [ ] Proper use of INumericOperations<T>

---

## Further Learning Resources

### Papers
1. **"Denoising Diffusion Probabilistic Models" (Ho et al., 2020)**: Foundation of diffusion models
2. **"Video Diffusion Models" (Ho et al., 2022)**: Extending diffusion to video
3. **"Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models" (2023)**: State-of-the-art video generation

### Concepts to Explore
- **Latent Diffusion**: Performing diffusion in a compressed latent space (faster)
- **Classifier-Free Guidance**: Technique to improve prompt adherence
- **Temporal Attention**: How models maintain consistency across time
- **Frame Interpolation**: Generating in-between frames for smoother video

### Example Pre-trained Models
- **ModelScope Text-to-Video**: Available on Hugging Face
- **Stable Video Diffusion**: Extension of Stable Diffusion for video
- **AnimateDiff**: Animation extensions for diffusion models

---

## Summary

You've implemented a **VideoGenerationModel** that:
1. Converts text prompts to embeddings using a text encoder
2. Initializes random noise as the starting point
3. Iteratively denoises the video over 50 steps
4. Maintains temporal coherence through spatiotemporal processing
5. Returns a sequence of coherent video frames

This is the foundation for text-to-video generation, one of the most exciting applications of generative AI. Well done!
