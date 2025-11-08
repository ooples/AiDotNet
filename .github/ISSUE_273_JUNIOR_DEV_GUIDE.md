# Issue #273: Junior Developer Implementation Guide - BLIP-2 Vision-Language Stack

## Understanding Vision-Language Models

### What is BLIP-2?

**BLIP-2** (Bootstrapping Language-Image Pre-training 2) is a state-of-the-art vision-language model that can:
- **Caption images:** Generate natural language descriptions
- **Answer questions about images:** Visual Question Answering (VQA)
- **Reason about visual content:** Understand relationships and context

**For Beginners:** Think of BLIP-2 as a person who can both see and talk:
- Show them a picture → They describe what they see
- Ask a question about the picture → They answer based on what they see
- It combines computer vision (seeing) with natural language processing (understanding/generating text)

### Three-Component Architecture

BLIP-2 has three main components:

```
┌─────────────┐     ┌──────────┐     ┌──────────────┐
│   Vision    │ --> │ Q-Former │ --> │   Language   │
│   Encoder   │     │ (Bridge) │     │    Model     │
│  (ViT/CNN)  │     │          │     │  (LLM/GPT)   │
└─────────────┘     └──────────┘     └──────────────┘
     ↓                   ↓                   ↓
  Image → Embeddings → Visual Queries → Text Generation
```

**Component roles:**

1. **Vision Encoder**: Converts images to visual features
   - Input: 224x224 image
   - Output: Grid of visual features (e.g., 196 patches, each 768-dim)

2. **Q-Former (Query Former)**: Bridges vision and language
   - Extracts relevant visual information with learnable queries
   - Compresses visual features into fixed number of tokens
   - Can interact with text questions

3. **Language Model**: Generates text responses
   - Takes visual query embeddings as "soft prompts"
   - Generates text autoregressively (one word at a time)

### Why Q-Former is Revolutionary

**The problem Q-Former solves:**
- Vision encoders produce **lots** of features (196+ patches)
- Language models work best with **compact** prompts (32-64 tokens)
- Direct connection is inefficient and loses information

**Q-Former solution:**
- Uses **learnable query tokens** (like asking specific questions to the image)
- Queries attend to image features and extract what's needed
- Outputs fixed number of visual tokens (e.g., 32 tokens, each 768-dim)

**Analogy:** Imagine you have a huge encyclopedia (vision features) and need to write a summary. Q-Former is like having smart questions that extract only the relevant facts you need.

### Two Operating Modes

**Mode 1: Image Captioning (Image-Only)**
```
Image → Vision Encoder → Q-Former (queries only attend to image)
                              ↓
                    Visual Tokens (32 x 768)
                              ↓
                    Language Model → "A dog playing in a park"
```

**Mode 2: Visual Question Answering (Image + Text)**
```
Image + Question → Vision Encoder → Q-Former (queries attend to both)
                                        ↓
                              Visual+Text Tokens
                                        ↓
                              Language Model → "The dog is brown"
```

---

## Existing Infrastructure to Use

### 1. Vision Encoder (Vision Transformer)

**File:** `src/NeuralNetworks/VisionTransformer.cs`

We already have:
- `VisionTransformer<T>` for image encoding
- Patch embedding layers
- Multi-head attention
- Output: Image features ready for Q-Former

**How BLIP-2 uses it:**
```csharp
// Use existing ViT to encode image
var visionEncoder = new VisionTransformer<T>(...);
var imageFeatures = visionEncoder.Predict(image);
// imageFeatures shape: [batch, num_patches, hidden_dim]
// Example: [1, 196, 768] for 14x14 patches
```

### 2. Attention Mechanisms

**File:** `src/NeuralNetworks/Layers/MultiHeadAttentionLayer.cs`

Q-Former uses cross-attention:
- **Query tokens** attend to **image features**
- **Query tokens** attend to **text tokens** (in VQA mode)

We can reuse `MultiHeadAttentionLayer<T>` for Q-Former attention blocks.

### 3. Transformer Architecture

**File:** `src/NeuralNetworks/Layers/TransformerEncoderLayer.cs`

Q-Former is essentially a stack of transformer layers with:
- Self-attention (queries interact with each other)
- Cross-attention (queries attend to image features)
- Feed-forward network

---

## Phase 1: Implementation Plan

### AC 1.1: Scaffolding the Blip2Model Wrapper (3 points)

**Step 1: Create Blip2Model.cs**

```csharp
// File: src/Models/Multimodal/Blip2Model.cs
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;
using System;
using System.Collections.Generic;

namespace AiDotNet.Models.Multimodal
{
    /// <summary>
    /// Implements BLIP-2 (Bootstrapping Language-Image Pre-training) for vision-language tasks.
    /// </summary>
    /// <typeparam name="T">The numeric type for computations (typically float or double).</typeparam>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> BLIP-2 is like giving a computer both eyes and a voice.
    /// It can look at images and:
    /// - Describe what it sees (image captioning)
    /// - Answer questions about the image (visual question answering)
    /// - Understand visual content in context
    ///
    /// The model has three parts:
    /// 1. Vision Encoder: Understands the image (like your eyes)
    /// 2. Q-Former: Asks smart questions about the image (like your visual processing)
    /// 3. Language Model: Generates text responses (like your speech)
    ///
    /// Example uses:
    /// - Accessibility: Describe images for visually impaired users
    /// - Content moderation: Understand what's in images automatically
    /// - Search: Find images by asking questions in natural language
    /// - Education: Interactive learning with visual content
    /// </para>
    /// </remarks>
    public class Blip2Model<T>
    {
        private readonly IOnnxModel<T> _visionEncoder;
        private readonly IOnnxModel<T> _qFormer;
        private readonly IOnnxModel<T> _languageModel;

        private readonly int _numQueryTokens;
        private readonly int _hiddenDim;

        protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        /// <summary>
        /// Creates a new BLIP-2 model wrapper with pre-trained ONNX components.
        /// </summary>
        /// <param name="visionEncoderPath">Path to the vision encoder ONNX model (typically ViT).</param>
        /// <param name="qFormerPath">Path to the Q-Former ONNX model (query transformer).</param>
        /// <param name="languageModelPath">Path to the language model ONNX model (typically OPT or FlanT5).</param>
        /// <param name="numQueryTokens">Number of learnable query tokens in Q-Former (default: 32).</param>
        /// <param name="hiddenDim">Hidden dimension of embeddings (default: 768).</param>
        /// <exception cref="ArgumentException">Thrown when paths are invalid or parameters are not positive.</exception>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This constructor loads three pre-trained neural networks:
        ///
        /// 1. **Vision Encoder**: Usually a Vision Transformer (ViT)
        ///    - Takes: 224x224 images
        ///    - Produces: Grid of visual features (e.g., 196 patches)
        ///
        /// 2. **Q-Former**: The "bridge" between vision and language
        ///    - Takes: Visual features + optional text
        ///    - Produces: Fixed number of visual query tokens (e.g., 32)
        ///    - This compression makes language generation efficient
        ///
        /// 3. **Language Model**: Generates text (like GPT)
        ///    - Takes: Visual query tokens as "soft prompts"
        ///    - Produces: Text, one word at a time
        ///
        /// Common BLIP-2 configurations:
        /// - BLIP-2 ViT-g + OPT-2.7B: 32 queries, 768 dims
        /// - BLIP-2 ViT-g + FlanT5-XL: 32 queries, 768 dims
        /// </para>
        /// </remarks>
        public Blip2Model(
            string visionEncoderPath,
            string qFormerPath,
            string languageModelPath,
            int numQueryTokens = 32,
            int hiddenDim = 768)
        {
            if (string.IsNullOrWhiteSpace(visionEncoderPath))
                throw new ArgumentException("Vision encoder path cannot be empty", nameof(visionEncoderPath));
            if (string.IsNullOrWhiteSpace(qFormerPath))
                throw new ArgumentException("Q-Former path cannot be empty", nameof(qFormerPath));
            if (string.IsNullOrWhiteSpace(languageModelPath))
                throw new ArgumentException("Language model path cannot be empty", nameof(languageModelPath));
            if (numQueryTokens <= 0)
                throw new ArgumentException("Number of query tokens must be positive", nameof(numQueryTokens));
            if (hiddenDim <= 0)
                throw new ArgumentException("Hidden dimension must be positive", nameof(hiddenDim));

            // Note: Replace with actual OnnxModel<T> when issue #280 is implemented
            _visionEncoder = LoadOnnxModel(visionEncoderPath);
            _qFormer = LoadOnnxModel(qFormerPath);
            _languageModel = LoadOnnxModel(languageModelPath);

            _numQueryTokens = numQueryTokens;
            _hiddenDim = hiddenDim;
        }

        private IOnnxModel<T> LoadOnnxModel(string path)
        {
            // TODO: Replace with actual OnnxModel<T> instantiation when #280 is complete
            throw new NotImplementedException("OnnxModel wrapper from issue #280 required");
        }
    }
}
```

**Step 2: Create unit test scaffold**

```csharp
// File: tests/UnitTests/Models/Blip2ModelTests.cs
using Xunit;
using AiDotNet.Models.Multimodal;
using AiDotNet.LinearAlgebra;
using Moq;

namespace AiDotNet.Tests.UnitTests.Models
{
    public class Blip2ModelTests
    {
        [Fact]
        public void Constructor_ValidPaths_CreatesInstance()
        {
            // Arrange
            string visionPath = "path/to/vision_encoder.onnx";
            string qformerPath = "path/to/qformer.onnx";
            string lmPath = "path/to/language_model.onnx";

            // Act & Assert
            var exception = Record.Exception(() =>
                new Blip2Model<double>(visionPath, qformerPath, lmPath));

            // Will throw NotImplementedException until #280 is done
            Assert.NotNull(exception);
        }

        [Theory]
        [InlineData(null, "qf.onnx", "lm.onnx")]
        [InlineData("", "qf.onnx", "lm.onnx")]
        [InlineData("ve.onnx", null, "lm.onnx")]
        [InlineData("ve.onnx", "", "lm.onnx")]
        [InlineData("ve.onnx", "qf.onnx", null)]
        [InlineData("ve.onnx", "qf.onnx", "")]
        public void Constructor_InvalidPaths_ThrowsArgumentException(
            string visionPath, string qformerPath, string lmPath)
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new Blip2Model<double>(visionPath, qformerPath, lmPath));
        }

        [Theory]
        [InlineData(0)]
        [InlineData(-1)]
        public void Constructor_InvalidNumQueryTokens_ThrowsArgumentException(int numTokens)
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new Blip2Model<double>("ve.onnx", "qf.onnx", "lm.onnx", numQueryTokens: numTokens));
        }
    }
}
```

---

### AC 1.2: Implement Image Captioning (8 points)

**Step 1: Add GenerateCaption method**

```csharp
/// <summary>
/// Generates a natural language caption describing the image.
/// </summary>
/// <param name="image">The input image tensor with shape [batch, channels, height, width].</param>
/// <param name="maxLength">Maximum caption length in tokens (default: 50).</param>
/// <param name="temperature">Sampling temperature for text generation (default: 1.0, higher = more random).</param>
/// <returns>A caption string describing the image content.</returns>
/// <exception cref="ArgumentNullException">Thrown when image is null.</exception>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Image captioning is like asking "What do you see in this picture?"
///
/// The process (BLIP-2 pipeline):
/// 1. **Vision Encoding**: Convert image to visual features
///    - Image (224x224) → Vision Encoder → Features (196 patches × 768 dims)
///
/// 2. **Query-Based Feature Extraction**: Use Q-Former to compress visual info
///    - Visual features → Q-Former → Query tokens (32 × 768 dims)
///    - Q-Former learns to "ask" the right questions about the image
///
/// 3. **Text Generation**: Language model generates caption
///    - Query tokens act as "soft prompts" (visual context)
///    - Generate words autoregressively: "A" → "A dog" → "A dog playing" → ...
///    - Stop at max length or end-of-sequence token
///
/// Example:
/// - Input: [Image of golden retriever in park]
/// - Output: "A golden retriever dog playing with a ball in a grassy park"
///
/// The `temperature` parameter controls creativity:
/// - 0.7: More conservative, safer captions
/// - 1.0: Balanced (default)
/// - 1.5: More creative, potentially less accurate
/// </para>
/// </remarks>
public string GenerateCaption(Tensor<T> image, int maxLength = 50, double temperature = 1.0)
{
    if (image == null)
        throw new ArgumentNullException(nameof(image));
    if (maxLength <= 0)
        throw new ArgumentException("Max length must be positive", nameof(maxLength));
    if (temperature <= 0)
        throw new ArgumentException("Temperature must be positive", nameof(temperature));

    // Step 1: Encode image to visual features
    var imageFeatures = EncodeImage(image);

    // Step 2: Use Q-Former to extract visual query tokens
    var queryTokens = ExtractVisualQueries(imageFeatures, textPrompt: null);

    // Step 3: Generate caption autoregressively
    var caption = GenerateText(queryTokens, maxLength, temperature);

    return caption;
}

/// <summary>
/// Encodes an image into visual feature representations.
/// </summary>
/// <param name="image">The input image tensor.</param>
/// <returns>Visual features tensor with shape [batch, num_patches, hidden_dim].</returns>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This step converts raw pixels into meaningful features.
///
/// What happens:
/// - Input: RGB image (3 × 224 × 224)
/// - Vision Encoder: Typically a Vision Transformer (ViT)
/// - Output: Patch features (1 × 196 × 768)
///   - 196 = 14×14 grid of patches (224/16 = 14)
///   - 768 = embedding dimension per patch
///
/// Each patch represents a 16×16 region of the image with rich semantic information.
/// </para>
/// </remarks>
private Tensor<T> EncodeImage(Tensor<T> image)
{
    // Preprocess image (resize, normalize)
    var preprocessed = PreprocessImage(image);

    // Run through vision encoder (ViT)
    var imageFeatures = _visionEncoder.Forward(preprocessed);

    return imageFeatures;
}

/// <summary>
/// Preprocesses image for BLIP-2 vision encoder.
/// </summary>
/// <param name="image">Raw input image.</param>
/// <returns>Preprocessed image ready for vision encoder.</returns>
/// <remarks>
/// <para>
/// BLIP-2 uses ImageNet normalization:
/// - Resize to 224×224
/// - Normalize: (pixel - mean) / std
/// - Mean: [0.485, 0.456, 0.406]
/// - Std: [0.229, 0.224, 0.225]
/// </para>
/// </remarks>
private Tensor<T> PreprocessImage(Tensor<T> image)
{
    const int targetSize = 224;
    var resized = ImageProcessor.Resize(image, targetSize, targetSize);

    // ImageNet normalization
    var meanR = NumOps.FromDouble(0.485);
    var meanG = NumOps.FromDouble(0.456);
    var meanB = NumOps.FromDouble(0.406);
    var stdR = NumOps.FromDouble(0.229);
    var stdG = NumOps.FromDouble(0.224);
    var stdB = NumOps.FromDouble(0.225);

    var normalized = new Tensor<T>(resized.Shape);
    int height = resized.Shape[0];
    int width = resized.Shape[1];

    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            normalized[h, w, 0] = NumOps.Divide(NumOps.Subtract(resized[h, w, 0], meanR), stdR);
            normalized[h, w, 1] = NumOps.Divide(NumOps.Subtract(resized[h, w, 1], meanG), stdG);
            normalized[h, w, 2] = NumOps.Divide(NumOps.Subtract(resized[h, w, 2], meanB), stdB);
        }
    }

    return normalized;
}

/// <summary>
/// Extracts visual query tokens using Q-Former.
/// </summary>
/// <param name="imageFeatures">Visual features from vision encoder.</param>
/// <param name="textPrompt">Optional text prompt for VQA (null for captioning).</param>
/// <returns>Query tokens representing compressed visual information.</returns>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Q-Former is the "bridge" between vision and language.
///
/// How it works:
/// 1. Start with learnable query tokens (e.g., 32 queries)
/// 2. Queries "attend" to image features (cross-attention)
///    - Like asking specific questions: "What color?" "What shape?" "Where?"
/// 3. Queries interact with each other (self-attention)
///    - Build coherent understanding
/// 4. Optional: Queries also attend to text (for VQA)
///
/// Output: Fixed number of query tokens (32 × 768)
/// - Much smaller than raw image features (196 × 768)
/// - Contains all relevant visual information for the task
///
/// Why compress? Language models work best with compact context.
/// 32 visual tokens + text is much better than 196 visual tokens + text.
/// </para>
/// </remarks>
private Tensor<T> ExtractVisualQueries(Tensor<T> imageFeatures, Tensor<T>? textPrompt)
{
    // Q-Former takes:
    // - Image features: [batch, 196, 768]
    // - Text tokens (optional): [batch, seq_len, 768]
    // Returns:
    // - Query tokens: [batch, 32, 768]

    if (textPrompt == null)
    {
        // Image-only mode (captioning)
        var queryTokens = _qFormer.Forward(imageFeatures);
        return queryTokens;
    }
    else
    {
        // Image + Text mode (VQA)
        // Q-Former attends to both modalities
        var combined = CombineImageAndText(imageFeatures, textPrompt);
        var queryTokens = _qFormer.Forward(combined);
        return queryTokens;
    }
}

/// <summary>
/// Generates text autoregressively from query tokens.
/// </summary>
/// <param name="queryTokens">Visual query tokens from Q-Former.</param>
/// <param name="maxLength">Maximum generation length.</param>
/// <param name="temperature">Sampling temperature.</param>
/// <returns>Generated text string.</returns>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Autoregressive generation means producing text one word at a time.
///
/// The process:
/// 1. Start with query tokens as "context" (visual information)
/// 2. Generate first word: P(word₁ | visual_context)
/// 3. Generate second word: P(word₂ | visual_context, word₁)
/// 4. Continue: P(wordₙ | visual_context, word₁, ..., wordₙ₋₁)
/// 5. Stop at max length or &lt;EOS&gt; (end-of-sequence) token
///
/// Temperature controls randomness:
/// - Low (0.5): Pick most likely word each time → safe, boring
/// - High (1.5): Sample from probability distribution → creative, risky
///
/// Example generation:
/// Visual context: [dog, grass, ball]
/// → "A" (most likely start)
/// → "A dog" (given "A", "dog" is likely)
/// → "A dog playing" (given "A dog", "playing" fits)
/// → "A dog playing with" (continue building)
/// → ... until complete sentence
/// </para>
/// </remarks>
private string GenerateText(Tensor<T> queryTokens, int maxLength, double temperature)
{
    var generatedTokens = new List<int>();
    const int bosToken = 1; // Beginning of sequence
    const int eosToken = 2; // End of sequence

    generatedTokens.Add(bosToken);

    for (int i = 0; i < maxLength; i++)
    {
        // Prepare input: query tokens + generated tokens so far
        var lmInput = PrepareLanguageModelInput(queryTokens, generatedTokens);

        // Get next token logits
        var logits = _languageModel.Forward(lmInput);

        // Sample next token with temperature
        int nextToken = SampleToken(logits, temperature);

        generatedTokens.Add(nextToken);

        // Stop if end-of-sequence
        if (nextToken == eosToken)
            break;
    }

    // Decode tokens to text
    string caption = DecodeTokens(generatedTokens);
    return caption;
}

/// <summary>
/// Samples a token from logits using temperature scaling.
/// </summary>
/// <param name="logits">Raw model outputs (unnormalized probabilities).</param>
/// <param name="temperature">Temperature for sampling.</param>
/// <returns>Sampled token ID.</returns>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Token sampling chooses the next word to generate.
///
/// Methods:
/// 1. **Greedy**: Always pick highest probability word
///    - Deterministic, safe, boring
///    - Can get stuck in repetitive loops
///
/// 2. **Temperature Sampling**: Scale probabilities, then sample
///    - Temperature = 1.0: Use raw probabilities
///    - Temperature &lt; 1.0: Make peaks sharper (more confident)
///    - Temperature &gt; 1.0: Flatten distribution (more random)
///
/// Formula: softmax(logits / temperature)
///
/// Example with temperature:
/// Raw logits: [2.0, 1.0, 0.5]
/// T=0.5: [0.7, 0.2, 0.1] ← More confident
/// T=1.0: [0.5, 0.3, 0.2] ← Original
/// T=2.0: [0.4, 0.3, 0.3] ← More uniform
/// </para>
/// </remarks>
private int SampleToken(Tensor<T> logits, double temperature)
{
    // Get last position logits (for next token)
    int vocabSize = logits.Shape[^1];
    var nextTokenLogits = new Vector<T>(vocabSize);
    for (int i = 0; i < vocabSize; i++)
    {
        nextTokenLogits[i] = logits[0, ^1, i]; // Last position
    }

    // Apply temperature scaling
    T tempT = NumOps.FromDouble(temperature);
    for (int i = 0; i < vocabSize; i++)
    {
        nextTokenLogits[i] = NumOps.Divide(nextTokenLogits[i], tempT);
    }

    // Convert to probabilities with softmax
    var probabilities = Softmax(nextTokenLogits);

    // Sample from distribution
    return SampleFromDistribution(probabilities);
}

/// <summary>
/// Samples an index from a probability distribution.
/// </summary>
private int SampleFromDistribution(Vector<T> probabilities)
{
    // Generate random number in [0, 1]
    double rand = Random.NextDouble();
    double cumulative = 0.0;

    // Sample using cumulative probability
    for (int i = 0; i < probabilities.Length; i++)
    {
        cumulative += Convert.ToDouble(probabilities[i]);
        if (rand <= cumulative)
            return i;
    }

    return probabilities.Length - 1; // Fallback
}

/// <summary>
/// Decodes token IDs back to text string.
/// </summary>
/// <param name="tokens">List of token IDs.</param>
/// <returns>Decoded text string.</returns>
private string DecodeTokens(List<int> tokens)
{
    // TODO: Implement proper tokenizer decoding
    // For now, placeholder showing structure

    // Would use tokenizer vocabulary:
    // tokens = [1, 250, 1929, 259, ...] (BOS, "A", "dog", "is", ...)
    // → "A dog is playing"

    throw new NotImplementedException("Requires tokenizer integration");
}

/// <summary>
/// Applies softmax to convert logits to probabilities.
/// </summary>
private Vector<T> Softmax(Vector<T> logits)
{
    // Find max for numerical stability
    T maxLogit = logits[0];
    for (int i = 1; i < logits.Length; i++)
    {
        if (NumOps.GreaterThan(logits[i], maxLogit))
            maxLogit = logits[i];
    }

    // Compute exp(logit - max)
    T sumExp = NumOps.Zero;
    var exps = new Vector<T>(logits.Length);
    for (int i = 0; i < logits.Length; i++)
    {
        T shifted = NumOps.Subtract(logits[i], maxLogit);
        exps[i] = NumOps.Exp(shifted);
        sumExp = NumOps.Add(sumExp, exps[i]);
    }

    // Normalize
    var probabilities = new Vector<T>(logits.Length);
    for (int i = 0; i < logits.Length; i++)
    {
        probabilities[i] = NumOps.Divide(exps[i], sumExp);
    }

    return probabilities;
}
```

**Step 2: Add unit tests for captioning**

```csharp
[Fact]
public void GenerateCaption_ValidImage_ReturnsNonEmptyString()
{
    // Arrange
    var mockVisionEncoder = new Mock<IOnnxModel<double>>();
    var mockQFormer = new Mock<IOnnxModel<double>>();
    var mockLanguageModel = new Mock<IOnnxModel<double>>();

    // Mock vision encoder: image → features
    var imageFeatures = new Tensor<double>(new[] { 1, 196, 768 });
    mockVisionEncoder.Setup(e => e.Forward(It.IsAny<Tensor<double>>()))
                     .Returns(imageFeatures);

    // Mock Q-Former: features → query tokens
    var queryTokens = new Tensor<double>(new[] { 1, 32, 768 });
    mockQFormer.Setup(q => q.Forward(It.IsAny<Tensor<double>>()))
               .Returns(queryTokens);

    // Mock language model: query tokens → text logits
    var logits = new Tensor<double>(new[] { 1, 50, 50257 }); // GPT vocab size
    mockLanguageModel.Setup(lm => lm.Forward(It.IsAny<Tensor<double>>()))
                     .Returns(logits);

    var model = new Blip2Model<double>(
        mockVisionEncoder.Object,
        mockQFormer.Object,
        mockLanguageModel.Object);

    var image = new Tensor<double>(new[] { 1, 3, 224, 224 });

    // Act
    var caption = model.GenerateCaption(image);

    // Assert
    Assert.NotNull(caption);
    Assert.NotEmpty(caption);

    // Verify pipeline was called
    mockVisionEncoder.Verify(e => e.Forward(It.IsAny<Tensor<double>>()), Times.Once);
    mockQFormer.Verify(q => q.Forward(It.IsAny<Tensor<double>>()), Times.Once);
    mockLanguageModel.Verify(lm => lm.Forward(It.IsAny<Tensor<double>>()), Times.AtLeastOnce);
}

[Fact]
public void GenerateCaption_InvalidMaxLength_ThrowsArgumentException()
{
    // Arrange
    var model = CreateMockBlip2Model();
    var image = new Tensor<double>(new[] { 1, 3, 224, 224 });

    // Act & Assert
    Assert.Throws<ArgumentException>(() => model.GenerateCaption(image, maxLength: 0));
    Assert.Throws<ArgumentException>(() => model.GenerateCaption(image, maxLength: -1));
}
```

---

### AC 1.3: Implement Visual Question Answering (VQA) (8 points)

**Step 1: Add Ask method**

```csharp
/// <summary>
/// Answers a question about the given image.
/// </summary>
/// <param name="image">The input image tensor.</param>
/// <param name="question">The question to answer about the image.</param>
/// <param name="maxLength">Maximum answer length in tokens (default: 30).</param>
/// <param name="temperature">Sampling temperature for generation (default: 1.0).</param>
/// <returns>An answer string responding to the question.</returns>
/// <exception cref="ArgumentNullException">Thrown when image or question is null.</exception>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Visual Question Answering (VQA) lets you ask questions about images.
///
/// The process (BLIP-2 VQA pipeline):
/// 1. **Vision Encoding**: Understand the image
///    - Image → Visual features (196 patches)
///
/// 2. **Text Encoding**: Understand the question
///    - Question: "What color is the dog?"
///    - Tokenize → [1, 753, 3195, 16, 5, 1929, 116, 2] (token IDs)
///    - Embed → Text features (N × 768)
///
/// 3. **Multimodal Fusion**: Q-Former combines both
///    - Query tokens attend to BOTH image features AND text features
///    - Learns to extract visual info relevant to the question
///    - Output: Query tokens encoding "brown dog" information
///
/// 4. **Answer Generation**: Language model produces answer
///    - Query tokens + question → Generate answer
///    - Output: "The dog is brown"
///
/// Example interactions:
/// - Q: "What is the dog doing?" → A: "Playing with a ball"
/// - Q: "Where is the dog?" → A: "In a park"
/// - Q: "How many dogs are there?" → A: "One"
///
/// The key difference from captioning:
/// - Captioning: Describe everything
/// - VQA: Focus only on what the question asks
/// </para>
/// </remarks>
public string Ask(Tensor<T> image, string question, int maxLength = 30, double temperature = 1.0)
{
    if (image == null)
        throw new ArgumentNullException(nameof(image));
    if (string.IsNullOrWhiteSpace(question))
        throw new ArgumentException("Question cannot be empty", nameof(question));
    if (maxLength <= 0)
        throw new ArgumentException("Max length must be positive", nameof(maxLength));
    if (temperature <= 0)
        throw new ArgumentException("Temperature must be positive", nameof(temperature));

    // Step 1: Encode image to visual features
    var imageFeatures = EncodeImage(image);

    // Step 2: Encode text question
    var textTokens = EncodeText(question);

    // Step 3: Use Q-Former to fuse image and text
    var queryTokens = ExtractVisualQueries(imageFeatures, textTokens);

    // Step 4: Generate answer autoregressively
    var answer = GenerateText(queryTokens, maxLength, temperature);

    return answer;
}

/// <summary>
/// Encodes text into token embeddings.
/// </summary>
/// <param name="text">The input text string.</param>
/// <returns>Text token embeddings.</returns>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Text encoding converts words into numbers.
///
/// Steps:
/// 1. **Tokenization**: Break into subwords
///    - "What color is the dog?" → ["What", "color", "is", "the", "dog", "?"]
///    - Each word → token ID: [753, 3195, 16, 5, 1929, 116]
///
/// 2. **Embedding**: Convert IDs to vectors
///    - Each ID → embedding vector (768 dims)
///    - "What" (753) → [0.12, -0.45, 0.89, ...]
///    - "color" (3195) → [-0.23, 0.67, 0.11, ...]
///
/// 3. **Position encoding**: Add position information
///    - So model knows word order matters
///
/// Output: [batch, sequence_length, hidden_dim]
/// Example: [1, 6, 768] for question with 6 tokens
/// </para>
/// </remarks>
private Tensor<T> EncodeText(string text)
{
    // Tokenize text
    var tokens = TokenizeText(text);

    // Embed tokens
    // In practice, this might be done inside Q-Former or a separate text encoder
    // For BLIP-2, text is often embedded within Q-Former itself

    return tokens; // Placeholder - actual implementation depends on model architecture
}

/// <summary>
/// Tokenizes text for BLIP-2 text encoder.
/// </summary>
/// <param name="text">Input text string.</param>
/// <returns>Tensor of token IDs.</returns>
private Tensor<T> TokenizeText(string text)
{
    // TODO: Implement proper tokenization
    // BLIP-2 typically uses BERT tokenizer or similar

    // Placeholder structure:
    const int maxLength = 32; // Max question length
    var tokenIds = new int[maxLength];

    // Would tokenize actual text here
    // Example: text = "What color is the dog?"
    // → tokens = [101, 2054, 3609, 2003, 1996, 3899, 102, 0, 0, ...]
    // (101 = CLS, 102 = SEP, 0 = PAD)

    var tokenTensor = new Tensor<T>(new[] { 1, maxLength });
    for (int i = 0; i < maxLength; i++)
    {
        tokenTensor[0, i] = NumOps.FromDouble(tokenIds[i]);
    }

    return tokenTensor;
}

/// <summary>
/// Combines image and text features for Q-Former.
/// </summary>
/// <param name="imageFeatures">Visual features from vision encoder.</param>
/// <param name="textFeatures">Text features from text encoding.</param>
/// <returns>Combined features for Q-Former.</returns>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This prepares inputs for Q-Former in VQA mode.
///
/// Q-Former needs to attend to both:
/// - Image features: What's visually present
/// - Text features: What the question is asking
///
/// Common approaches:
/// 1. **Concatenate**: [image_feats; text_feats] → single sequence
/// 2. **Separate attention**: Q-Former has cross-attention to each
/// 3. **Interleaved**: Alternate image and text tokens
///
/// The Q-Former queries then extract information from both modalities
/// to form a comprehensive answer.
/// </para>
/// </remarks>
private Tensor<T> CombineImageAndText(Tensor<T> imageFeatures, Tensor<T> textFeatures)
{
    // Implementation depends on Q-Former architecture
    // Common approach: concatenate along sequence dimension

    int batchSize = imageFeatures.Shape[0];
    int imageSeqLen = imageFeatures.Shape[1];
    int textSeqLen = textFeatures.Shape[1];
    int hiddenDim = imageFeatures.Shape[2];

    var combined = new Tensor<T>(new[] { batchSize, imageSeqLen + textSeqLen, hiddenDim });

    // Copy image features
    for (int b = 0; b < batchSize; b++)
    {
        for (int i = 0; i < imageSeqLen; i++)
        {
            for (int d = 0; d < hiddenDim; d++)
            {
                combined[b, i, d] = imageFeatures[b, i, d];
            }
        }
    }

    // Copy text features
    for (int b = 0; b < batchSize; b++)
    {
        for (int i = 0; i < textSeqLen; i++)
        {
            for (int d = 0; d < hiddenDim; d++)
            {
                combined[b, imageSeqLen + i, d] = textFeatures[b, i, d];
            }
        }
    }

    return combined;
}
```

**Step 2: Add unit tests for VQA**

```csharp
[Fact]
public void Ask_ValidImageAndQuestion_ReturnsNonEmptyAnswer()
{
    // Arrange
    var model = CreateMockBlip2Model();
    var image = new Tensor<double>(new[] { 1, 3, 224, 224 });
    string question = "What color is the dog?";

    // Act
    var answer = model.Ask(image, question);

    // Assert
    Assert.NotNull(answer);
    Assert.NotEmpty(answer);
}

[Theory]
[InlineData(null)]
[InlineData("")]
[InlineData("   ")]
public void Ask_InvalidQuestion_ThrowsArgumentException(string question)
{
    // Arrange
    var model = CreateMockBlip2Model();
    var image = new Tensor<double>(new[] { 1, 3, 224, 224 });

    // Act & Assert
    Assert.Throws<ArgumentException>(() => model.Ask(image, question));
}

[Fact]
public void Ask_NullImage_ThrowsArgumentNullException()
{
    // Arrange
    var model = CreateMockBlip2Model();

    // Act & Assert
    Assert.Throws<ArgumentNullException>(() => model.Ask(null, "question"));
}

[Fact]
public void Ask_CallsCorrectPipeline()
{
    // Arrange
    var mockVisionEncoder = new Mock<IOnnxModel<double>>();
    var mockQFormer = new Mock<IOnnxModel<double>>();
    var mockLanguageModel = new Mock<IOnnxModel<double>>();

    // Setup mocks
    mockVisionEncoder.Setup(e => e.Forward(It.IsAny<Tensor<double>>()))
                     .Returns(new Tensor<double>(new[] { 1, 196, 768 }));
    mockQFormer.Setup(q => q.Forward(It.IsAny<Tensor<double>>()))
               .Returns(new Tensor<double>(new[] { 1, 32, 768 }));
    mockLanguageModel.Setup(lm => lm.Forward(It.IsAny<Tensor<double>>()))
                     .Returns(new Tensor<double>(new[] { 1, 30, 50257 }));

    var model = new Blip2Model<double>(
        mockVisionEncoder.Object,
        mockQFormer.Object,
        mockLanguageModel.Object);

    var image = new Tensor<double>(new[] { 1, 3, 224, 224 });

    // Act
    var answer = model.Ask(image, "What is this?");

    // Assert
    // Vision encoder should be called once
    mockVisionEncoder.Verify(e => e.Forward(It.IsAny<Tensor<double>>()), Times.Once);

    // Q-Former should be called once (with combined image + text)
    mockQFormer.Verify(q => q.Forward(It.IsAny<Tensor<double>>()), Times.Once);

    // Language model should be called multiple times (autoregressive generation)
    mockLanguageModel.Verify(lm => lm.Forward(It.IsAny<Tensor<double>>()), Times.AtLeastOnce);
}
```

---

## Phase 2: Integration Testing

### AC 2.2: Integration Test with Real ONNX Models (8 points)

```csharp
// File: tests/IntegrationTests/Models/Blip2ModelIntegrationTests.cs
using Xunit;
using AiDotNet.Models.Multimodal;
using AiDotNet.LinearAlgebra;
using System;
using System.IO;

namespace AiDotNet.Tests.IntegrationTests.Models
{
    /// <summary>
    /// Integration tests for BLIP-2 model using real ONNX files.
    /// </summary>
    public class Blip2ModelIntegrationTests : IDisposable
    {
        private readonly string _modelDir;
        private readonly string _visionEncoderPath;
        private readonly string _qFormerPath;
        private readonly string _languageModelPath;
        private bool _modelsAvailable;

        public Blip2ModelIntegrationTests()
        {
            _modelDir = Path.Combine(Path.GetTempPath(), "blip2_models");
            _visionEncoderPath = Path.Combine(_modelDir, "blip2_vision_encoder.onnx");
            _qFormerPath = Path.Combine(_modelDir, "blip2_qformer.onnx");
            _languageModelPath = Path.Combine(_modelDir, "blip2_opt_2_7b.onnx");

            _modelsAvailable = File.Exists(_visionEncoderPath) &&
                             File.Exists(_qFormerPath) &&
                             File.Exists(_languageModelPath);
        }

        [Fact(Skip = "Requires manual model download")]
        public void GenerateCaption_DogImage_GeneratesRelevantCaption()
        {
            if (!_modelsAvailable)
            {
                Console.WriteLine("BLIP-2 models not found. Download from:");
                Console.WriteLine("https://huggingface.co/Salesforce/blip2-opt-2.7b");
                return;
            }

            // Arrange
            var model = new Blip2Model<float>(
                _visionEncoderPath,
                _qFormerPath,
                _languageModelPath,
                numQueryTokens: 32,
                hiddenDim: 768);

            var image = LoadTestImage("test_images/dog_on_beach.jpg");

            // Act
            var caption = model.GenerateCaption(image, maxLength: 50);

            // Assert
            Assert.NotNull(caption);
            Assert.NotEmpty(caption);

            // Caption should mention relevant concepts
            string lowerCaption = caption.ToLower();
            bool hasRelevantContent = lowerCaption.Contains("dog") ||
                                     lowerCaption.Contains("beach") ||
                                     lowerCaption.Contains("sand") ||
                                     lowerCaption.Contains("animal");

            Assert.True(hasRelevantContent,
                $"Caption '{caption}' doesn't mention relevant content");

            Console.WriteLine($"Generated caption: {caption}");
        }

        [Fact(Skip = "Requires manual model download")]
        public void Ask_DogColorQuestion_AnswersCorrectly()
        {
            if (!_modelsAvailable) return;

            // Arrange
            var model = new Blip2Model<float>(
                _visionEncoderPath,
                _qFormerPath,
                _languageModelPath);

            var image = LoadTestImage("test_images/golden_retriever.jpg");
            string question = "What color is the dog?";

            // Act
            var answer = model.Ask(image, question, maxLength: 20);

            // Assert
            Assert.NotNull(answer);
            Assert.NotEmpty(answer);

            // Answer should mention a plausible color
            string lowerAnswer = answer.ToLower();
            bool hasColor = lowerAnswer.Contains("golden") ||
                           lowerAnswer.Contains("yellow") ||
                           lowerAnswer.Contains("brown") ||
                           lowerAnswer.Contains("tan");

            Assert.True(hasColor,
                $"Answer '{answer}' doesn't mention a color");

            Console.WriteLine($"Q: {question}");
            Console.WriteLine($"A: {answer}");
        }

        [Fact(Skip = "Requires manual model download")]
        public void Ask_MultipleQuestions_AllGetRelevantAnswers()
        {
            if (!_modelsAvailable) return;

            // Arrange
            var model = new Blip2Model<float>(
                _visionEncoderPath,
                _qFormerPath,
                _languageModelPath);

            var image = LoadTestImage("test_images/scene_with_multiple_objects.jpg");

            var questions = new[]
            {
                "How many people are in the image?",
                "What is the weather like?",
                "What are the people doing?"
            };

            // Act & Assert
            foreach (var question in questions)
            {
                var answer = model.Ask(image, question);

                Assert.NotNull(answer);
                Assert.NotEmpty(answer);

                Console.WriteLine($"Q: {question}");
                Console.WriteLine($"A: {answer}");
                Console.WriteLine();
            }
        }

        private Tensor<float> LoadTestImage(string path)
        {
            // Load image from file and convert to tensor
            // Would use ImageSharp or similar library

            throw new NotImplementedException("Requires image loading library");
        }

        public void Dispose()
        {
            // Cleanup if needed
        }
    }
}
```

---

## Common Pitfalls and Best Practices

### 1. Autoregressive Generation Requires Careful State Management

**❌ Wrong:**
```csharp
// Regenerating all tokens from scratch each iteration
for (int i = 0; i < maxLength; i++)
{
    var output = model.Forward(allTokens); // Inefficient!
    nextToken = SampleToken(output);
}
```

**✅ Better (with caching):**
```csharp
// Cache past key-value states in transformer
var cache = InitializeCache();
for (int i = 0; i < maxLength; i++)
{
    var output = model.ForwardWithCache(lastToken, cache);
    nextToken = SampleToken(output);
}
```

### 2. Temperature Sampling Edge Cases

**❌ Wrong:**
```csharp
// Can cause numerical issues
var scaled = logits / temperature; // Division by zero if temp = 0
```

**✅ Correct:**
```csharp
if (temperature < 1e-8)
{
    // Greedy sampling
    return ArgMax(logits);
}
else
{
    // Temperature sampling
    var scaled = logits / temperature;
    return SampleFromSoftmax(scaled);
}
```

### 3. Handle Variable-Length Sequences

**❌ Wrong:**
```csharp
// Assumes fixed caption length
var tokens = new List<int>(maxLength);
```

**✅ Correct:**
```csharp
// Stop at EOS token OR max length
var tokens = new List<int>();
while (tokens.Count < maxLength && lastToken != eosToken)
{
    // Generate next token
}
```

### 4. Q-Former Attention Masks

**Important:** Q-Former needs attention masks to:
- Prevent queries from attending to padding tokens
- Separate image attention from text attention (in VQA mode)

```csharp
// Create attention mask for combined input
var attentionMask = CreateAttentionMask(
    imageSeqLen: 196,
    textSeqLen: 32,
    paddingLen: 10);
```

### 5. Memory Management for Large Models

BLIP-2 models can be very large (~3GB+ for OPT-2.7B):
- Load models lazily
- Unload when not in use
- Consider model quantization (FP16, INT8)

---

## Testing Strategy

### Unit Tests (Mocked)
- Test each pipeline stage independently
- Verify correct shapes throughout pipeline
- Test error handling (null inputs, invalid parameters)
- Test generation stopping conditions

### Integration Tests (Real Models)
- Test end-to-end captioning
- Test various VQA question types
- Test with different image contents
- Verify semantic correctness (not exact matches)

### Performance Tests
```csharp
[Fact]
public void GenerateCaption_Performance_ReasonablyFast()
{
    var sw = Stopwatch.StartNew();
    var caption = model.GenerateCaption(image);
    sw.Stop();

    // Should generate in under 5 seconds (hardware dependent)
    Assert.True(sw.Elapsed.TotalSeconds < 5);
}
```

---

## Next Steps After Implementation

1. **Add beam search decoding** for better caption quality
2. **Implement nucleus (top-p) sampling** for more diverse generation
3. **Add conversation mode** for multi-turn VQA
4. **Support image-text retrieval** (find images matching text queries)
5. **Fine-tune Q-Former** for domain-specific tasks
6. **Add vision-language pretraining** from scratch

---

## Summary

This guide covered:
- ✅ Understanding BLIP-2 three-component architecture
- ✅ Implementing image captioning with Q-Former
- ✅ Implementing visual question answering
- ✅ Autoregressive text generation with temperature sampling
- ✅ Multimodal fusion strategies
- ✅ Comprehensive testing for vision-language models

**Key concepts:**
- **Q-Former:** Bridge between vision and language modalities
- **Learnable queries:** Extract task-relevant visual information
- **Autoregressive generation:** Produce text one token at a time
- **Temperature sampling:** Control generation randomness/creativity
- **Multimodal fusion:** Combine image and text understanding

**Dependencies:**
- Issue #280: OnnxModel wrapper (prerequisite)
- Issue #272: CLIP understanding (reference for encoders)
- Issue #330: Image preprocessing (required for vision encoder)

**For beginners:** Start with understanding the three components separately, then connect them. Test captioning first (simpler), then add VQA (requires text encoding). Focus on the data flow: Image → Features → Queries → Text.
