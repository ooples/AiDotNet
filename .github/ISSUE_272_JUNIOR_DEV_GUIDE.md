# Issue #272: Junior Developer Implementation Guide - CLIP-Style Text/Image Encoders

## Understanding Multimodal Embeddings

### What is CLIP?

**CLIP** (Contrastive Language-Image Pre-training) is a neural network trained to understand images and text together in a shared embedding space.

**For Beginners:** Think of CLIP like a universal translator between images and words:
- It can take a photo and convert it to a vector of numbers (image embedding)
- It can take text and convert it to a vector of numbers (text embedding)
- If the photo matches the text, their vectors will be **close together** in mathematical space
- If they don't match, their vectors will be **far apart**

**Real-world analogy:** Imagine you have a magical dictionary where:
- Every image gets a code number (like "horse-12345")
- Every description gets a code number (like "animal-12346")
- Similar things get similar codes
- You can compare codes to see if an image matches a description!

### How Contrastive Learning Works

CLIP uses **contrastive learning** to align image and text representations:

1. **Training pairs:** Show the model millions of image-caption pairs
2. **Positive pairs:** Match image embeddings with their correct captions (pull close)
3. **Negative pairs:** Separate image embeddings from wrong captions (push apart)
4. **Result:** After training, similar concepts cluster together in embedding space

**Key insight:** You don't need explicit labels! The model learns from natural language descriptions.

### Zero-Shot Classification Magic

Once CLIP is trained, you can classify images **without ever training on those specific classes**:

```
Image: [photo of a dog]
Candidates: ["a photo of a cat", "a photo of a dog", "a photo of a bird"]

1. Embed image → vector [0.2, 0.8, 0.1, ...]
2. Embed each text → vectors
   - "cat" → [0.1, 0.3, 0.9, ...]
   - "dog" → [0.19, 0.81, 0.09, ...] ← VERY CLOSE!
   - "bird" → [0.7, 0.1, 0.2, ...]
3. Calculate similarity (cosine similarity / dot product)
4. Apply softmax → probabilities [5%, 90%, 5%]
5. Answer: "dog" with 90% confidence!
```

### Why L2 Normalization Matters

**L2 normalization** makes all embedding vectors have length = 1 (unit vectors).

**Why normalize?**
- **Cosine similarity becomes dot product:** Much faster to compute!
- **Focus on direction, not magnitude:** Only the angle between vectors matters
- **Stable comparisons:** All vectors are on the same scale

**Math:**
```
Original vector: [3, 4] → magnitude = sqrt(3² + 4²) = 5
Normalized: [3/5, 4/5] = [0.6, 0.8] → magnitude = 1
```

**In code:**
```csharp
Vector<T> normalized = embedding.Normalize(); // Divide by magnitude
```

---

## Existing Infrastructure to Use

### 1. Vision Transformer (ViT) for Image Encoding

**File:** `src/NeuralNetworks/VisionTransformer.cs`

CLIP's image encoder is typically a Vision Transformer. We already have:
- `VisionTransformer<T>` class with patch embedding
- `PatchEmbeddingLayer<T>` to convert images → patches
- `MultiHeadAttentionLayer<T>` for transformer blocks
- Classification head can be replaced with embedding projection

**Key insight:** Remove the final classification layer, use the CLS token output as the image embedding!

### 2. Attention Mechanisms

**File:** `src/NeuralNetworks/Layers/MultiHeadAttentionLayer.cs`

Already implemented:
- Multi-head self-attention
- Query, Key, Value projections
- Attention score calculation with softmax

### 3. Image Preprocessing

**File:** `.github/ISSUE_330_JUNIOR_DEV_GUIDE.md`

Guide shows how to:
- Resize images with bilinear interpolation
- Normalize pixel values
- Handle RGB/grayscale conversion

### 4. Tensor Operations

**Files:**
- `src/LinearAlgebra/Tensor.cs` - Multi-dimensional arrays
- `src/LinearAlgebra/Vector.cs` - 1D vectors with normalization
- `src/LinearAlgebra/Matrix.cs` - 2D matrices

Already supports:
- Element-wise operations
- Matrix multiplication
- L2 normalization: `vector.Normalize()`

---

## Phase 1: Implementation Plan

### AC 1.1: Scaffolding the ClipModel Wrapper (3 points)

**Step 1: Create the folder structure**

```bash
# Create multimodal folder
mkdir src/Models/Multimodal
```

**Step 2: Create ClipModel.cs**

```csharp
// File: src/Models/Multimodal/ClipModel.cs
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;
using System;

namespace AiDotNet.Models.Multimodal
{
    /// <summary>
    /// Implements CLIP (Contrastive Language-Image Pre-training) model for multimodal embeddings.
    /// </summary>
    /// <typeparam name="T">The numeric type for computations (typically float or double).</typeparam>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> CLIP is like a universal translator between images and text.
    /// It converts both images and text into vectors of numbers (embeddings) in a shared space.
    /// If an image matches a text description, their embeddings will be close together.
    /// This allows tasks like:
    /// - Zero-shot image classification: "Is this a cat or dog?" without training on those classes
    /// - Image search: Find images matching a text query
    /// - Text search: Find text matching an image
    /// </para>
    /// <para>
    /// This implementation wraps two ONNX models:
    /// 1. Image encoder: Converts images to embeddings (typically a Vision Transformer)
    /// 2. Text encoder: Converts text to embeddings (typically a Transformer)
    /// </para>
    /// </remarks>
    public class ClipModel<T>
    {
        private readonly IOnnxModel<T> _imageEncoder;
        private readonly IOnnxModel<T> _textEncoder;
        private readonly int _embeddingDim;
        protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        /// <summary>
        /// Creates a new CLIP model wrapper with pre-trained ONNX encoders.
        /// </summary>
        /// <param name="imageEncoderPath">Path to the image encoder ONNX model file.</param>
        /// <param name="textEncoderPath">Path to the text encoder ONNX model file.</param>
        /// <param name="embeddingDim">The dimension of the embedding vectors (default: 512 for CLIP ViT-B/32).</param>
        /// <exception cref="ArgumentException">Thrown when paths are invalid or embedding dimension is not positive.</exception>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This constructor loads two pre-trained models:
        /// - Image encoder: Understands visual content
        /// - Text encoder: Understands language
        /// Both produce vectors of the same size (embeddingDim) so they can be compared.
        ///
        /// Common CLIP configurations:
        /// - CLIP ViT-B/32: 512-dimensional embeddings
        /// - CLIP ViT-B/16: 512-dimensional embeddings
        /// - CLIP ViT-L/14: 768-dimensional embeddings
        /// </para>
        /// </remarks>
        public ClipModel(string imageEncoderPath, string textEncoderPath, int embeddingDim = 512)
        {
            if (string.IsNullOrWhiteSpace(imageEncoderPath))
                throw new ArgumentException("Image encoder path cannot be empty", nameof(imageEncoderPath));
            if (string.IsNullOrWhiteSpace(textEncoderPath))
                throw new ArgumentException("Text encoder path cannot be empty", nameof(textEncoderPath));
            if (embeddingDim <= 0)
                throw new ArgumentException("Embedding dimension must be positive", nameof(embeddingDim));

            // Note: Replace with actual OnnxModel<T> when issue #280 is implemented
            _imageEncoder = LoadOnnxModel(imageEncoderPath);
            _textEncoder = LoadOnnxModel(textEncoderPath);
            _embeddingDim = embeddingDim;
        }

        private IOnnxModel<T> LoadOnnxModel(string path)
        {
            // TODO: Replace with actual OnnxModel<T> instantiation when #280 is complete
            throw new NotImplementedException("OnnxModel wrapper from issue #280 required");
        }
    }
}
```

**Step 3: Create unit test scaffold**

```csharp
// File: tests/UnitTests/Models/ClipModelTests.cs
using Xunit;
using AiDotNet.Models.Multimodal;
using AiDotNet.LinearAlgebra;
using Moq;

namespace AiDotNet.Tests.UnitTests.Models
{
    public class ClipModelTests
    {
        [Fact]
        public void Constructor_ValidPaths_CreatesInstance()
        {
            // Arrange
            string imagePath = "path/to/image_encoder.onnx";
            string textPath = "path/to/text_encoder.onnx";

            // Act & Assert
            var exception = Record.Exception(() => new ClipModel<double>(imagePath, textPath));

            // Will throw NotImplementedException until #280 is done
            Assert.NotNull(exception);
        }

        [Theory]
        [InlineData(null, "text.onnx")]
        [InlineData("", "text.onnx")]
        [InlineData("image.onnx", null)]
        [InlineData("image.onnx", "")]
        public void Constructor_InvalidPaths_ThrowsArgumentException(string imagePath, string textPath)
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new ClipModel<double>(imagePath, textPath));
        }

        [Fact]
        public void Constructor_InvalidEmbeddingDim_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new ClipModel<double>("image.onnx", "text.onnx", embeddingDim: -1));
        }
    }
}
```

---

### AC 1.2: Implement Image and Text Embedding Methods (8 points)

**Step 1: Add GetImageEmbedding method**

```csharp
/// <summary>
/// Generates a normalized embedding for an input image.
/// </summary>
/// <param name="image">The input image tensor with shape [batch, channels, height, width].</param>
/// <returns>A normalized embedding vector with magnitude 1.</returns>
/// <exception cref="ArgumentNullException">Thrown when image is null.</exception>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This method converts an image into a vector of numbers (embedding).
///
/// The process:
/// 1. Preprocess: Resize and normalize the image for the model
/// 2. Encode: Run through the image encoder (Vision Transformer)
/// 3. Normalize: Scale the vector to length 1 (L2 normalization)
///
/// Why normalize? So we can compare embeddings using simple dot products.
/// All vectors have the same "length", only their direction matters.
///
/// Example:
/// - Input: 224x224 RGB image of a dog
/// - Output: [0.12, -0.45, 0.89, ...] (512 numbers, magnitude = 1)
/// </para>
/// </remarks>
public Vector<T> GetImageEmbedding(Tensor<T> image)
{
    if (image == null)
        throw new ArgumentNullException(nameof(image));

    // Step 1: Preprocess image
    var preprocessed = PreprocessImage(image);

    // Step 2: Run through image encoder
    var rawEmbedding = _imageEncoder.Forward(preprocessed);

    // Step 3: Extract embedding vector (typically from output layer or CLS token)
    var embedding = ExtractEmbeddingVector(rawEmbedding);

    // Step 4: L2 normalize to unit length
    return NormalizeEmbedding(embedding);
}

/// <summary>
/// Preprocesses an image for CLIP image encoder.
/// </summary>
/// <param name="image">The raw input image.</param>
/// <returns>Preprocessed image tensor ready for encoding.</returns>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Preprocessing prepares the image for the neural network.
///
/// Standard CLIP preprocessing:
/// 1. Resize to 224x224 (or model's expected size)
/// 2. Normalize pixels: (pixel - mean) / std
///    - Mean: [0.48145466, 0.4578275, 0.40821073] (ImageNet statistics)
///    - Std: [0.26862954, 0.26130258, 0.27577711]
/// 3. Convert to CHW format: [channels, height, width]
/// </para>
/// </remarks>
private Tensor<T> PreprocessImage(Tensor<T> image)
{
    // Resize to expected input size (typically 224x224)
    const int targetSize = 224;
    var resized = ImageProcessor.Resize(image, targetSize, targetSize);

    // Normalize with ImageNet statistics
    var meanR = NumOps.FromDouble(0.48145466);
    var meanG = NumOps.FromDouble(0.4578275);
    var meanB = NumOps.FromDouble(0.40821073);
    var stdR = NumOps.FromDouble(0.26862954);
    var stdG = NumOps.FromDouble(0.26130258);
    var stdB = NumOps.FromDouble(0.27577711);

    var normalized = new Tensor<T>(resized.Shape);
    int height = resized.Shape[0];
    int width = resized.Shape[1];

    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            // Normalize each channel
            var r = resized[h, w, 0];
            var g = resized[h, w, 1];
            var b = resized[h, w, 2];

            normalized[h, w, 0] = NumOps.Divide(NumOps.Subtract(r, meanR), stdR);
            normalized[h, w, 1] = NumOps.Divide(NumOps.Subtract(g, meanG), stdG);
            normalized[h, w, 2] = NumOps.Divide(NumOps.Subtract(b, meanB), stdB);
        }
    }

    return normalized;
}

/// <summary>
/// Extracts the embedding vector from the encoder output.
/// </summary>
/// <param name="encoderOutput">Raw output from ONNX encoder.</param>
/// <returns>The embedding vector.</returns>
private Vector<T> ExtractEmbeddingVector(Tensor<T> encoderOutput)
{
    // For Vision Transformer: typically extract the CLS token (first token)
    // Output shape is usually [batch, sequence_length, embedding_dim]
    // We want the [0, 0, :] slice (first batch, CLS token, all dimensions)

    int embeddingDim = encoderOutput.Shape[^1]; // Last dimension
    var embedding = new Vector<T>(embeddingDim);

    for (int i = 0; i < embeddingDim; i++)
    {
        embedding[i] = encoderOutput[0, 0, i]; // Batch 0, Token 0 (CLS), Dim i
    }

    return embedding;
}

/// <summary>
/// Normalizes an embedding vector to unit length (L2 normalization).
/// </summary>
/// <param name="embedding">The unnormalized embedding.</param>
/// <returns>The normalized embedding with magnitude 1.</returns>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> L2 normalization scales a vector to have length 1.
///
/// Why? So cosine similarity becomes a simple dot product:
/// - cos(θ) = (A · B) / (|A| × |B|)
/// - If |A| = |B| = 1, then cos(θ) = A · B
///
/// Formula: normalized = v / ||v||
/// Where ||v|| = sqrt(v₁² + v₂² + ... + vₙ²)
/// </para>
/// </remarks>
private Vector<T> NormalizeEmbedding(Vector<T> embedding)
{
    // Calculate L2 norm (magnitude)
    T sumOfSquares = NumOps.Zero;
    for (int i = 0; i < embedding.Length; i++)
    {
        sumOfSquares = NumOps.Add(sumOfSquares, NumOps.Multiply(embedding[i], embedding[i]));
    }
    T magnitude = NumOps.Sqrt(sumOfSquares);

    // Avoid division by zero
    if (NumOps.LessThanOrEqual(magnitude, NumOps.FromDouble(1e-8)))
    {
        throw new InvalidOperationException("Cannot normalize zero-magnitude vector");
    }

    // Divide each component by magnitude
    var normalized = new Vector<T>(embedding.Length);
    for (int i = 0; i < embedding.Length; i++)
    {
        normalized[i] = NumOps.Divide(embedding[i], magnitude);
    }

    return normalized;
}
```

**Step 2: Add GetTextEmbedding method**

```csharp
/// <summary>
/// Generates a normalized embedding for input text.
/// </summary>
/// <param name="text">The input text string.</param>
/// <returns>A normalized embedding vector with magnitude 1.</returns>
/// <exception cref="ArgumentNullException">Thrown when text is null.</exception>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This method converts text into a vector of numbers (embedding).
///
/// The process:
/// 1. Tokenize: Break text into tokens (like words or subwords)
/// 2. Encode: Run through text encoder (Transformer)
/// 3. Pool: Combine token embeddings (typically use CLS token or mean pooling)
/// 4. Normalize: Scale to unit length
///
/// Example:
/// - Input: "a photo of a dog"
/// - Tokens: [49406, 320, 1125, 539, 320, 1929, 49407] (token IDs)
/// - Output: [0.15, -0.42, 0.87, ...] (512 numbers, magnitude = 1)
/// </para>
/// </remarks>
public Vector<T> GetTextEmbedding(string text)
{
    if (text == null)
        throw new ArgumentNullException(nameof(text));

    // Step 1: Tokenize text
    var tokens = TokenizeText(text);

    // Step 2: Run through text encoder
    var rawEmbedding = _textEncoder.Forward(tokens);

    // Step 3: Extract embedding vector (typically CLS token or mean pool)
    var embedding = ExtractEmbeddingVector(rawEmbedding);

    // Step 4: L2 normalize to unit length
    return NormalizeEmbedding(embedding);
}

/// <summary>
/// Tokenizes text for CLIP text encoder.
/// </summary>
/// <param name="text">The input text string.</param>
/// <returns>Tensor of token IDs ready for encoding.</returns>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Tokenization converts text into numbers the model understands.
///
/// CLIP uses byte-pair encoding (BPE) tokenization:
/// 1. Split text into subwords: "running" → ["run", "ning"]
/// 2. Convert to IDs: ["run", "ning"] → [1234, 5678]
/// 3. Add special tokens: [CLS] text [SEP] → [49406, 1234, 5678, 49407]
/// 4. Pad to fixed length: [49406, 1234, 5678, 49407, 0, 0, ...]
///
/// Max length is typically 77 tokens for CLIP.
/// </para>
/// </remarks>
private Tensor<T> TokenizeText(string text)
{
    // TODO: Implement proper BPE tokenization
    // For now, this is a placeholder showing the expected structure

    const int maxLength = 77; // CLIP's context length
    const int clsToken = 49406;
    const int sepToken = 49407;
    const int padToken = 0;

    // This would use a real tokenizer (like tiktoken for CLIP)
    // For demonstration:
    var tokenIds = new int[maxLength];
    tokenIds[0] = clsToken;

    // TODO: Tokenize actual text here
    // Example: var tokens = _tokenizer.Encode(text);
    // Copy tokens to tokenIds[1...n-1]

    // Add separator at end
    tokenIds[^1] = sepToken;

    // Convert to tensor
    var tokenTensor = new Tensor<T>(new[] { 1, maxLength }); // [batch=1, sequence_length]
    for (int i = 0; i < maxLength; i++)
    {
        tokenTensor[0, i] = NumOps.FromDouble(tokenIds[i]);
    }

    return tokenTensor;
}
```

**Step 3: Add unit tests**

```csharp
[Fact]
public void GetImageEmbedding_ValidImage_ReturnsNormalizedVector()
{
    // Arrange
    var mockImageEncoder = new Mock<IOnnxModel<double>>();
    var image = new Tensor<double>(new[] { 1, 3, 224, 224 }); // Batch=1, RGB, 224x224

    // Mock encoder returns a raw embedding
    var mockOutput = new Tensor<double>(new[] { 1, 1, 512 });
    for (int i = 0; i < 512; i++)
        mockOutput[0, 0, i] = i * 0.01; // Non-normalized values

    mockImageEncoder.Setup(e => e.Forward(It.IsAny<Tensor<double>>()))
                    .Returns(mockOutput);

    var model = new ClipModel<double>(mockImageEncoder.Object, null, 512);

    // Act
    var embedding = model.GetImageEmbedding(image);

    // Assert
    Assert.NotNull(embedding);
    Assert.Equal(512, embedding.Length);

    // Check normalization: magnitude should be 1
    double magnitude = 0;
    for (int i = 0; i < embedding.Length; i++)
        magnitude += embedding[i] * embedding[i];
    magnitude = Math.Sqrt(magnitude);

    Assert.InRange(magnitude, 0.99, 1.01); // Allow small floating point error
}

[Fact]
public void GetTextEmbedding_ValidText_ReturnsNormalizedVector()
{
    // Arrange
    var mockTextEncoder = new Mock<IOnnxModel<double>>();
    string text = "a photo of a dog";

    // Mock encoder returns a raw embedding
    var mockOutput = new Tensor<double>(new[] { 1, 77, 512 }); // Batch=1, SeqLen=77, Dim=512
    for (int i = 0; i < 512; i++)
        mockOutput[0, 0, i] = i * 0.01;

    mockTextEncoder.Setup(e => e.Forward(It.IsAny<Tensor<double>>()))
                   .Returns(mockOutput);

    var model = new ClipModel<double>(null, mockTextEncoder.Object, 512);

    // Act
    var embedding = model.GetTextEmbedding(text);

    // Assert
    Assert.NotNull(embedding);
    Assert.Equal(512, embedding.Length);

    // Check normalization
    double magnitude = 0;
    for (int i = 0; i < embedding.Length; i++)
        magnitude += embedding[i] * embedding[i];
    magnitude = Math.Sqrt(magnitude);

    Assert.InRange(magnitude, 0.99, 1.01);
}
```

---

### AC 1.3: Implement Zero-Shot Classification (5 points)

**Step 1: Add ZeroShotClassify method**

```csharp
/// <summary>
/// Classifies an image against multiple text descriptions without training.
/// </summary>
/// <param name="image">The input image to classify.</param>
/// <param name="classCandidates">List of text descriptions representing possible classes.</param>
/// <returns>Dictionary mapping each class to its probability (0-1, sum to 1).</returns>
/// <exception cref="ArgumentNullException">Thrown when image or classCandidates is null.</exception>
/// <exception cref="ArgumentException">Thrown when classCandidates is empty.</exception>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Zero-shot classification lets you classify images into categories
/// the model was never explicitly trained on. It's "zero-shot" because you need zero examples!
///
/// How it works:
/// 1. Convert the image to an embedding (vector)
/// 2. Convert each text description to an embedding (vector)
/// 3. Measure similarity: How close is the image to each description?
/// 4. Convert similarities to probabilities using softmax
///
/// Example:
/// - Image: [photo of golden retriever]
/// - Candidates: ["a cat", "a dog", "a bird"]
/// - Similarities: [0.2, 0.9, 0.1] → dog is closest!
/// - Probabilities: [0.05, 0.90, 0.05] → 90% confident it's a dog
///
/// This is powerful because you can create new categories just by writing descriptions!
/// No retraining needed.
/// </para>
/// </remarks>
public Dictionary<string, double> ZeroShotClassify(Tensor<T> image, List<string> classCandidates)
{
    if (image == null)
        throw new ArgumentNullException(nameof(image));
    if (classCandidates == null)
        throw new ArgumentNullException(nameof(classCandidates));
    if (classCandidates.Count == 0)
        throw new ArgumentException("Must provide at least one class candidate", nameof(classCandidates));

    // Step 1: Get image embedding
    var imageEmbedding = GetImageEmbedding(image);

    // Step 2: Get text embeddings for all candidates
    var textEmbeddings = new List<Vector<T>>();
    foreach (var candidate in classCandidates)
    {
        textEmbeddings.Add(GetTextEmbedding(candidate));
    }

    // Step 3: Calculate cosine similarities (dot products, since vectors are normalized)
    var similarities = new List<T>();
    foreach (var textEmbedding in textEmbeddings)
    {
        T similarity = CalculateCosineSimilarity(imageEmbedding, textEmbedding);
        similarities.Add(similarity);
    }

    // Step 4: Apply softmax to convert similarities to probabilities
    var probabilities = Softmax(similarities);

    // Step 5: Build result dictionary
    var result = new Dictionary<string, double>();
    for (int i = 0; i < classCandidates.Count; i++)
    {
        result[classCandidates[i]] = Convert.ToDouble(probabilities[i]);
    }

    return result;
}

/// <summary>
/// Calculates cosine similarity between two vectors.
/// </summary>
/// <param name="a">First vector.</param>
/// <param name="b">Second vector.</param>
/// <returns>Cosine similarity in range [-1, 1].</returns>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Cosine similarity measures how similar two vectors are.
///
/// - Value of 1: Vectors point in same direction (very similar)
/// - Value of 0: Vectors are perpendicular (unrelated)
/// - Value of -1: Vectors point in opposite directions (very different)
///
/// Formula: cos(θ) = (A · B) / (||A|| × ||B||)
/// But since our vectors are already normalized (||A|| = ||B|| = 1),
/// it simplifies to: cos(θ) = A · B (dot product)
/// </para>
/// </remarks>
private T CalculateCosineSimilarity(Vector<T> a, Vector<T> b)
{
    if (a.Length != b.Length)
        throw new ArgumentException("Vectors must have same length");

    // For normalized vectors, cosine similarity = dot product
    T dotProduct = NumOps.Zero;
    for (int i = 0; i < a.Length; i++)
    {
        dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(a[i], b[i]));
    }

    return dotProduct;
}

/// <summary>
/// Applies softmax function to convert scores to probabilities.
/// </summary>
/// <param name="scores">List of similarity scores.</param>
/// <returns>List of probabilities that sum to 1.</returns>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Softmax converts arbitrary numbers into probabilities.
///
/// Properties:
/// - All outputs are between 0 and 1
/// - All outputs sum to exactly 1
/// - Larger inputs get larger probabilities
///
/// Formula: softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
///
/// Example:
/// - Input: [2.0, 1.0, 0.1]
/// - Exponentials: [7.39, 2.72, 1.11]
/// - Sum: 11.22
/// - Output: [0.66, 0.24, 0.10] ← Notice they sum to 1!
///
/// We use a temperature parameter to control "confidence":
/// - Higher temp → more even distribution
/// - Lower temp → more confident (winner takes all)
/// </para>
/// </remarks>
private List<T> Softmax(List<T> scores)
{
    // For numerical stability, subtract max value
    T maxScore = scores[0];
    foreach (var score in scores)
    {
        if (NumOps.GreaterThan(score, maxScore))
            maxScore = score;
    }

    // Calculate exp(score - max)
    var exponents = new List<T>();
    T sumExp = NumOps.Zero;
    foreach (var score in scores)
    {
        T shiftedScore = NumOps.Subtract(score, maxScore);
        T expValue = NumOps.Exp(shiftedScore);
        exponents.Add(expValue);
        sumExp = NumOps.Add(sumExp, expValue);
    }

    // Divide by sum to get probabilities
    var probabilities = new List<T>();
    foreach (var expValue in exponents)
    {
        probabilities.Add(NumOps.Divide(expValue, sumExp));
    }

    return probabilities;
}
```

**Step 2: Add comprehensive unit tests**

```csharp
[Fact]
public void ZeroShotClassify_ValidInputs_ReturnsProbabilities()
{
    // Arrange
    var mockImageEncoder = new Mock<IOnnxModel<double>>();
    var mockTextEncoder = new Mock<IOnnxModel<double>>();

    // Setup mock image embedding: [1, 0, 0, ...] (512 dims)
    var imageOutput = new Tensor<double>(new[] { 1, 1, 512 });
    imageOutput[0, 0, 0] = 1.0; // Normalized: first component = 1
    mockImageEncoder.Setup(e => e.Forward(It.IsAny<Tensor<double>>()))
                    .Returns(imageOutput);

    // Setup mock text embeddings with varying similarities
    int callCount = 0;
    mockTextEncoder.Setup(e => e.Forward(It.IsAny<Tensor<double>>()))
                   .Returns(() =>
                   {
                       var output = new Tensor<double>(new[] { 1, 77, 512 });
                       // First candidate: similar to image [0.9, 0, 0, ...]
                       // Second candidate: less similar [0.5, 0, 0, ...]
                       // Third candidate: dissimilar [0, 1, 0, ...]
                       if (callCount == 0)
                           output[0, 0, 0] = 0.9;
                       else if (callCount == 1)
                           output[0, 0, 0] = 0.5;
                       else
                           output[0, 0, 1] = 1.0; // Orthogonal
                       callCount++;
                       return output;
                   });

    var model = new ClipModel<double>(mockImageEncoder.Object, mockTextEncoder.Object, 512);
    var image = new Tensor<double>(new[] { 1, 3, 224, 224 });
    var candidates = new List<string> { "a dog", "a cat", "a bird" };

    // Act
    var result = model.ZeroShotClassify(image, candidates);

    // Assert
    Assert.Equal(3, result.Count);
    Assert.True(result.ContainsKey("a dog"));
    Assert.True(result.ContainsKey("a cat"));
    Assert.True(result.ContainsKey("a bird"));

    // Probabilities should sum to 1
    double sum = result.Values.Sum();
    Assert.InRange(sum, 0.99, 1.01);

    // First candidate should have highest probability
    Assert.True(result["a dog"] > result["a cat"]);
    Assert.True(result["a dog"] > result["a bird"]);
}

[Fact]
public void ZeroShotClassify_SingleCandidate_Returns100Percent()
{
    // Arrange
    var model = CreateMockClipModel();
    var image = new Tensor<double>(new[] { 1, 3, 224, 224 });
    var candidates = new List<string> { "only option" };

    // Act
    var result = model.ZeroShotClassify(image, candidates);

    // Assert
    Assert.Single(result);
    Assert.InRange(result["only option"], 0.99, 1.01); // Should be ~1.0
}

[Fact]
public void ZeroShotClassify_NullImage_ThrowsArgumentNullException()
{
    // Arrange
    var model = CreateMockClipModel();
    var candidates = new List<string> { "a", "b" };

    // Act & Assert
    Assert.Throws<ArgumentNullException>(() => model.ZeroShotClassify(null, candidates));
}

[Fact]
public void ZeroShotClassify_EmptyCandidates_ThrowsArgumentException()
{
    // Arrange
    var model = CreateMockClipModel();
    var image = new Tensor<double>(new[] { 1, 3, 224, 224 });
    var candidates = new List<string>();

    // Act & Assert
    Assert.Throws<ArgumentException>(() => model.ZeroShotClassify(image, candidates));
}
```

---

## Phase 2: Integration Testing

### AC 2.2: Integration Test with Real ONNX Models (8 points)

**Step 1: Create integration test with model download**

```csharp
// File: tests/IntegrationTests/Models/ClipModelIntegrationTests.cs
using Xunit;
using AiDotNet.Models.Multimodal;
using AiDotNet.LinearAlgebra;
using System;
using System.IO;
using System.Net.Http;

namespace AiDotNet.Tests.IntegrationTests.Models
{
    /// <summary>
    /// Integration tests for CLIP model using real ONNX files.
    /// </summary>
    /// <remarks>
    /// These tests require downloading CLIP ONNX models (~400MB each).
    /// Run the setup script first: tests/IntegrationTests/Setup/download_clip_models.sh
    /// </remarks>
    public class ClipModelIntegrationTests : IDisposable
    {
        private readonly string _modelDir;
        private readonly string _imageEncoderPath;
        private readonly string _textEncoderPath;
        private bool _modelsAvailable;

        public ClipModelIntegrationTests()
        {
            _modelDir = Path.Combine(Path.GetTempPath(), "clip_models");
            _imageEncoderPath = Path.Combine(_modelDir, "clip_vit_b32_image.onnx");
            _textEncoderPath = Path.Combine(_modelDir, "clip_vit_b32_text.onnx");

            _modelsAvailable = File.Exists(_imageEncoderPath) && File.Exists(_textEncoderPath);
        }

        [Fact(Skip = "Requires manual model download - run download_clip_models.sh first")]
        public void ZeroShotClassify_AstronautImage_CorrectlyIdentifies()
        {
            // Skip if models not available
            if (!_modelsAvailable)
            {
                // Log instructions
                Console.WriteLine("CLIP models not found. To run this test:");
                Console.WriteLine("1. Download models from: https://huggingface.co/openai/clip-vit-base-patch32");
                Console.WriteLine($"2. Place in: {_modelDir}");
                return;
            }

            // Arrange
            var model = new ClipModel<float>(_imageEncoderPath, _textEncoderPath, embeddingDim: 512);

            // Load test image: astronaut riding a horse
            var image = LoadTestImage("test_images/astronaut_horse.jpg");

            var prompts = new List<string>
            {
                "an astronaut riding a horse",
                "a doctor checking a patient",
                "a chef cooking in a kitchen"
            };

            // Act
            var result = model.ZeroShotClassify(image, prompts);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(3, result.Count);

            // Find the highest probability class
            var maxClass = result.OrderByDescending(kvp => kvp.Value).First();

            // The astronaut prompt should have highest probability
            Assert.Equal("an astronaut riding a horse", maxClass.Key);

            // Should be reasonably confident (>50%)
            Assert.True(maxClass.Value > 0.5,
                $"Expected confidence >50%, got {maxClass.Value:P2}");

            // Log results for debugging
            Console.WriteLine("Classification results:");
            foreach (var kvp in result.OrderByDescending(x => x.Value))
            {
                Console.WriteLine($"  {kvp.Key}: {kvp.Value:P2}");
            }
        }

        [Fact(Skip = "Requires manual model download")]
        public void GetImageEmbedding_MultipleImages_SimilarImagesHaveHighSimilarity()
        {
            if (!_modelsAvailable) return;

            // Arrange
            var model = new ClipModel<float>(_imageEncoderPath, _textEncoderPath, 512);

            // Load two similar images (both dogs) and one different (cat)
            var dog1 = LoadTestImage("test_images/dog1.jpg");
            var dog2 = LoadTestImage("test_images/dog2.jpg");
            var cat = LoadTestImage("test_images/cat.jpg");

            // Act
            var emb_dog1 = model.GetImageEmbedding(dog1);
            var emb_dog2 = model.GetImageEmbedding(dog2);
            var emb_cat = model.GetImageEmbedding(cat);

            // Calculate similarities
            double sim_dog1_dog2 = CosineSimilarity(emb_dog1, emb_dog2);
            double sim_dog1_cat = CosineSimilarity(emb_dog1, emb_cat);

            // Assert
            // Two dogs should be more similar than dog and cat
            Assert.True(sim_dog1_dog2 > sim_dog1_cat,
                $"Dog-Dog similarity ({sim_dog1_dog2:F3}) should be > Dog-Cat similarity ({sim_dog1_cat:F3})");

            Console.WriteLine($"Dog1-Dog2 similarity: {sim_dog1_dog2:F3}");
            Console.WriteLine($"Dog1-Cat similarity: {sim_dog1_cat:F3}");
        }

        private Tensor<float> LoadTestImage(string path)
        {
            // Load image from file and convert to tensor
            // This would use an actual image loading library (like ImageSharp)

            // For demonstration:
            // 1. Read image file
            // 2. Decode to RGB pixels
            // 3. Convert to Tensor<float> with shape [1, 3, height, width]

            throw new NotImplementedException("Requires image loading library");
        }

        private double CosineSimilarity(Vector<float> a, Vector<float> b)
        {
            double dotProduct = 0;
            for (int i = 0; i < a.Length; i++)
            {
                dotProduct += a[i] * b[i];
            }
            return dotProduct; // Already normalized vectors
        }

        public void Dispose()
        {
            // Cleanup if needed
        }
    }
}
```

**Step 2: Create model download script**

```bash
#!/bin/bash
# File: tests/IntegrationTests/Setup/download_clip_models.sh
#
# Downloads pre-trained CLIP ONNX models for integration testing
# Models are ~400MB each, so this is optional for unit tests

set -e

MODEL_DIR="${TMPDIR:-/tmp}/clip_models"
mkdir -p "$MODEL_DIR"

echo "Downloading CLIP ViT-B/32 ONNX models..."
echo "This will download ~800MB of models"
echo ""

# Option 1: Download from Hugging Face
# You would use the actual URLs here
echo "Note: Update this script with actual ONNX model URLs"
echo "Example sources:"
echo "  - https://huggingface.co/openai/clip-vit-base-patch32"
echo "  - https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus"

echo ""
echo "Models should be placed in: $MODEL_DIR"
echo "Expected files:"
echo "  - clip_vit_b32_image.onnx"
echo "  - clip_vit_b32_text.onnx"
```

---

## Common Pitfalls and Best Practices

### 1. Always Normalize Embeddings

**❌ Wrong:**
```csharp
var embedding = _encoder.Forward(input);
return ExtractVector(embedding); // Not normalized!
```

**✅ Correct:**
```csharp
var embedding = _encoder.Forward(input);
var vector = ExtractVector(embedding);
return NormalizeEmbedding(vector); // L2 normalized
```

**Why:** Cosine similarity only works correctly with unit vectors.

### 2. Use NumOps for All Arithmetic

**❌ Wrong:**
```csharp
T result = a + b; // Doesn't work for generic T
```

**✅ Correct:**
```csharp
T result = NumOps.Add(a, b); // Works for any numeric type
```

### 3. Handle Numerical Stability in Softmax

**❌ Wrong:**
```csharp
var exps = scores.Select(s => Math.Exp(s)).ToList(); // Can overflow!
```

**✅ Correct:**
```csharp
// Subtract max for stability
T max = scores.Max();
var exps = scores.Select(s => Exp(s - max)).ToList();
```

**Why:** `exp(large_number)` can overflow. Subtracting the max prevents this.

### 4. Validate Tensor Shapes

**❌ Wrong:**
```csharp
var embedding = encoderOutput[0, 0]; // Might crash!
```

**✅ Correct:**
```csharp
if (encoderOutput.Shape.Length != 3)
    throw new ArgumentException("Expected 3D tensor");
var embedding = encoderOutput[0, 0, i];
```

### 5. Test with Multiple Numeric Types

```csharp
[Theory]
[InlineData(typeof(double))]
[InlineData(typeof(float))]
public void GetImageEmbedding_DifferentTypes_Works(Type numericType)
{
    // Test with both double and float
    // Ensures NumOps abstraction works correctly
}
```

---

## Testing Strategy

### Unit Tests (Mocked ONNX Models)

**Test coverage targets:**
- Constructor validation (paths, embedding dimension)
- Image preprocessing (resize, normalization)
- Text tokenization (edge cases: empty, very long)
- Embedding extraction and normalization
- Cosine similarity calculation
- Softmax probabilities sum to 1
- Zero-shot classification logic

**Use mocks to:**
- Avoid downloading large ONNX models
- Control encoder outputs for predictable tests
- Test error conditions

### Integration Tests (Real ONNX Models)

**Test realistic scenarios:**
- Load actual CLIP models
- Classify real images
- Verify semantic similarity (dogs vs cats)
- Test with various image sizes and formats

**Make tests optional:**
- Use `[Fact(Skip = "...")]` attribute
- Check for model files before running
- Provide clear instructions for setup

### Performance Tests

```csharp
[Fact]
public void GetImageEmbedding_Performance_UnderThreshold()
{
    // Ensure encoding is reasonably fast
    var image = CreateTestImage();

    var sw = Stopwatch.StartNew();
    var embedding = model.GetImageEmbedding(image);
    sw.Stop();

    // Should encode in under 100ms (with mock)
    Assert.True(sw.ElapsedMilliseconds < 100);
}
```

---

## Next Steps After Implementation

1. **Add image-image similarity search**
2. **Implement batch processing** for multiple images/texts
3. **Add caching** for frequently used text embeddings
4. **Optimize preprocessing** (GPU acceleration, SIMD)
5. **Support additional CLIP variants** (ViT-L, ViT-H)
6. **Integrate with vector databases** for large-scale search

---

## Summary

This guide covered:
- ✅ Understanding CLIP and contrastive learning
- ✅ Implementing image and text encoders
- ✅ L2 normalization for cosine similarity
- ✅ Zero-shot classification with softmax
- ✅ Comprehensive testing strategies
- ✅ Integration with existing AiDotNet infrastructure

**Key concepts:**
- **Shared embedding space:** Images and text in the same vector space
- **Contrastive learning:** Pull similar pairs together, push dissimilar apart
- **Zero-shot:** Classify without training examples
- **L2 normalization:** Unit vectors enable efficient similarity

**Dependencies:**
- Issue #280: OnnxModel wrapper (prerequisite)
- Issue #330: Image preprocessing (reference guide)
- Existing: VisionTransformer, MultiHeadAttention, Tensor operations

**For beginners:** Start with the unit tests using mocks. Understand the flow: Image → Preprocess → Encode → Normalize → Compare → Classify. Once that works, add real ONNX integration.
