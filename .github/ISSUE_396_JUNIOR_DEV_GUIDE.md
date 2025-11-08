# Junior Developer Implementation Guide: Issue #396
## Vision-Language Models (CLIP, BLIP, Flamingo)

### Overview
This guide will walk you through implementing Vision-Language Models for AiDotNet. These models bridge the gap between visual and textual understanding, enabling tasks like image-text retrieval, visual question answering, and zero-shot image classification. They power applications from search engines to AI assistants.

---

## Understanding Vision-Language Models

### What Are Vision-Language Models?

Vision-Language Models (VLMs) learn joint representations of images and text, understanding how visual content relates to language.

**Core Idea**: Map images and text into a shared embedding space where semantically similar pairs are close together.

**Real-World Analogy**:
- Imagine a universal translator that can understand both pictures and words
- It learns that a photo of a dog and the word "dog" represent the same concept
- It can then connect "a photo of a puppy" with "adorable young canine" even if it never saw that exact pairing

### Why Vision-Language Models Are Revolutionary

1. **Zero-Shot Transfer**: Classify images into categories never seen during training
2. **Multimodal Understanding**: Connect visual and linguistic concepts
3. **Flexible Tasks**: One model for retrieval, classification, captioning, VQA
4. **Natural Language Interface**: Use text prompts instead of fixed class labels

### Key Architectures

#### 1. CLIP (Contrastive Language-Image Pre-training)

**Paper**: "Learning Transferable Visual Models From Natural Language Supervision" (OpenAI, 2021)

**Key Innovation**: Train image and text encoders jointly using contrastive learning

**Architecture**:
```
Image → Image Encoder (Vision Transformer) → Image Embedding (512-dim)
                                                ↓
                                         Cosine Similarity
                                                ↓
Text → Text Encoder (Transformer) → Text Embedding (512-dim)
```

**Training Objective (Contrastive Loss)**:
```
Given a batch of (image, text) pairs:

1. Encode all images: I₁, I₂, ..., Iₙ → image embeddings
2. Encode all texts: T₁, T₂, ..., Tₙ → text embeddings
3. Compute similarity matrix: S[i,j] = cosine_sim(I_i, T_j)
4. Maximize diagonal (matching pairs), minimize off-diagonal

Loss = CrossEntropy(S, labels) + CrossEntropy(S^T, labels)

Where labels = [0, 1, 2, ..., N-1] (diagonal)
```

**Why it works**:
- Learns from 400M image-text pairs from the internet
- Natural language supervision is more scalable than manual labels
- Embeddings capture rich semantic relationships

**Zero-Shot Classification**:
```
To classify an image into ["cat", "dog", "bird"]:

1. Encode image: I_embed
2. Encode text prompts: "a photo of a cat", "a photo of a dog", "a photo of a bird"
3. Compute similarities: [sim(I, T_cat), sim(I, T_dog), sim(I, T_bird)]
4. Predict: argmax(similarities)

No fine-tuning needed!
```

#### 2. BLIP (Bootstrapping Language-Image Pre-training)

**Paper**: "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation" (Salesforce, 2022)

**Key Innovation**: Unified model for both understanding and generation tasks

**Architecture**:
```
Image Encoder (ViT) → Visual Features
                         ↓
    ┌────────────────────┴────────────────────┐
    ↓                                         ↓
Image-Text Contrastive (ITC)    Image-Grounded Text Generation (Decoder)
    ↓                                         ↓
Image-Text Matching (ITM)          Captioning / VQA
    ↓
Fine-grained alignment
```

**Three Training Objectives**:

1. **ITC (Image-Text Contrastive)**:
   - Same as CLIP
   - Learn global alignment

2. **ITM (Image-Text Matching)**:
   - Binary classification: matching or not
   - Cross-attention between vision and language
   - Learn fine-grained alignment

3. **LM (Language Modeling)**:
   - Generate captions given images
   - Uses causal decoder with cross-attention to visual features

**CapFilt (Captioning and Filtering)**:
- Generate synthetic captions for web images
- Filter noisy captions using ITM model
- Bootstrap better training data

**Why it matters**:
- Outperforms CLIP on retrieval tasks
- Can generate captions and answer questions
- Handles noisy web data better

#### 3. Flamingo

**Paper**: "Flamingo: a Visual Language Model for Few-Shot Learning" (DeepMind, 2022)

**Key Innovation**: Few-shot learning with in-context examples

**Architecture**:
```
Vision Encoder (frozen NFNet)
    ↓
Perceiver Resampler (compress visual features)
    ↓
    ├─→ Cross-attention to Language Model
    ↓
Large Language Model (70B Chinchilla, frozen)
    ├─→ Gated cross-attention layers (learned)
    ↓
Generated Text
```

**Key Components**:

1. **Perceiver Resampler**:
   - Compress variable-resolution images to fixed number of tokens
   - Uses cross-attention to learned queries
   - Reduces visual tokens from ~10K to ~64

2. **Gated Cross-Attention**:
   - Inserted between LM layers
   - Attend to visual features
   - Gating allows smooth incorporation of vision
   - tanh(α) · CrossAttn(vision) + LM_hidden

3. **Interleaved Training Data**:
   - Mix text and images in sequences
   - Example: [img1, text1, img2, text2, question, answer]
   - Learn from context

**Few-Shot Learning**:
```
Input:
[img: cat] "This is a cat"
[img: dog] "This is a dog"  ← Few-shot examples
[img: bird] "This is a ___"  ← Query

Output: "bird"

The model learns from the in-context examples!
```

**Why it matters**:
- State-of-the-art few-shot performance
- Can handle multi-image inputs
- Natural dialogue-style interactions

---

## Core Concepts

### 1. Contrastive Learning

**Goal**: Learn embeddings where similar pairs are close, dissimilar pairs are far

**InfoNCE Loss** (used in CLIP):
```
For a positive pair (image i, text i):

Loss_i = -log(exp(sim(I_i, T_i) / τ) / Σⱼ exp(sim(I_i, T_j) / τ))

Where:
- sim(a, b) = cosine_similarity(a, b)
- τ is temperature (typically 0.07)
- Sum is over all texts in the batch
```

**Intuition**:
- Numerator: Similarity of matching pair
- Denominator: Sum of similarities to all texts (including non-matching)
- Maximizing this ratio pulls matching pairs together, pushes others apart

**Symmetric Loss**:
```
Total Loss = (Loss_image_to_text + Loss_text_to_image) / 2
```

### 2. Cross-Modal Attention

**Standard self-attention**: Query, Key, Value all from same modality

**Cross-attention**: Query from one modality, Key/Value from another

```
Text queries attend to visual features:

Q = W_q · text_embeddings
K = W_k · visual_features
V = W_v · visual_features

Attention(Q, K, V) = softmax(QK^T / sqrt(d)) · V
```

**Why it works**:
- Text queries can "look at" specific image regions
- Enables fine-grained alignment
- Used in ITM (image-text matching) and VQA

### 3. Vision Encoder Options

**Choices**:

1. **Convolutional Networks** (ResNet):
   - Traditional choice
   - Good for low-level features
   - Spatially structured

2. **Vision Transformers** (ViT):
   - CLIP's choice
   - Better for global context
   - More scalable

3. **Hybrid** (ConvNet + ViT):
   - Best of both worlds
   - Used in some recent models

**Output**: Spatial feature map or sequence of patch embeddings

### 4. Text Encoder Options

**Choices**:

1. **Transformer Encoder** (BERT-style):
   - Bidirectional context
   - Good for understanding
   - Used in CLIP, BLIP

2. **Transformer Decoder** (GPT-style):
   - Causal (left-to-right)
   - Good for generation
   - Used in BLIP's language modeling head

**Output**: Sequence of token embeddings or pooled representation

### 5. Embedding Space Design

**Key Properties**:

1. **Shared Dimensionality**: Images and text project to same dimension (e.g., 512)
2. **Normalized**: Embeddings are L2-normalized (unit length)
3. **Cosine Similarity**: Distance metric in embedding space
4. **Temperature Scaling**: Control sharpness of similarity distribution

**Why normalize**:
- Cosine similarity only depends on angle, not magnitude
- More stable training
- Easier to interpret (range: -1 to 1)

---

## Architecture Overview

### File Structure
```
src/
├── Interfaces/
│   ├── IVisionLanguageModel.cs       # Base VLM interface
│   ├── IContrastiveModel.cs          # For CLIP-style models
│   ├── IImageTextEncoder.cs          # Dual encoder interface
│   ├── IMultimodalEncoder.cs         # Cross-attention encoder
│   └── IVisualQuestionAnswering.cs   # VQA interface
├── Models/
│   └── VisionLanguage/
│       ├── VisionLanguageModelBase.cs    # Base class
│       ├── CLIPModel.cs                  # CLIP implementation
│       ├── BLIPModel.cs                  # BLIP implementation
│       └── FlamingoModel.cs              # Flamingo implementation
├── VisionLanguage/
│   ├── Encoders/
│   │   ├── ImageEncoder.cs          # Vision encoder (ViT or ResNet)
│   │   ├── TextEncoder.cs           # Text encoder (Transformer)
│   │   └── MultimodalEncoder.cs     # Cross-attention fusion
│   ├── Contrastive/
│   │   ├── ContrastiveLoss.cs       # InfoNCE loss
│   │   └── SimilarityMatrix.cs      # Batch similarity computation
│   ├── CrossAttention/
│   │   ├── CrossAttentionLayer.cs   # Cross-modal attention
│   │   └── GatedCrossAttention.cs   # Flamingo-style gating
│   └── Projection/
│       ├── ProjectionHead.cs        # Project to shared embedding space
│       └── PerceiverResampler.cs    # Compress visual features
```

### Class Hierarchy
```
IVisionLanguageModel<T>
    ↓
VisionLanguageModelBase<T> (abstract)
    ├── CLIPModel<T>         # Contrastive pre-training
    ├── BLIPModel<T>         # Unified understanding + generation
    └── FlamingoModel<T>     # Few-shot learning

IContrastiveModel<T>
    └── CLIPModel<T>

IVisualQuestionAnswering<T>
    ├── BLIPModel<T>
    └── FlamingoModel<T>
```

---

## Step-by-Step Implementation

### Step 1: Core Interfaces

#### File: `src/Interfaces/IVisionLanguageModel.cs`

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a vision-language model that understands both images and text.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// Vision-Language Models bridge visual and textual understanding by:
///
/// 1. **Encoding**: Convert images and text into numerical representations
/// 2. **Alignment**: Learn relationships between visual and linguistic concepts
/// 3. **Tasks**: Enable image-text retrieval, classification, VQA, captioning
///
/// Key capabilities:
/// - Zero-shot classification using text descriptions
/// - Image-text retrieval (find images matching text, or vice versa)
/// - Visual question answering
/// - Image captioning
///
/// These models learn from large-scale image-text pairs from the web,
/// understanding natural language descriptions of visual content.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public interface IVisionLanguageModel<T>
{
    /// <summary>
    /// Encodes an image into a fixed-dimensional embedding.
    /// </summary>
    /// <param name="image">Input image tensor [batch, channels, height, width].</param>
    /// <returns>Image embedding [batch, embedding_dim].</returns>
    Tensor<T> EncodeImage(Tensor<T> image);

    /// <summary>
    /// Encodes text into a fixed-dimensional embedding.
    /// </summary>
    /// <param name="text">Input text tokens [batch, sequence_length].</param>
    /// <returns>Text embedding [batch, embedding_dim].</returns>
    Tensor<T> EncodeText(int[][] text);

    /// <summary>
    /// Computes similarity between image and text embeddings.
    /// </summary>
    /// <param name="imageEmbedding">Image embeddings [batch_images, dim].</param>
    /// <param name="textEmbedding">Text embeddings [batch_texts, dim].</param>
    /// <returns>Similarity matrix [batch_images, batch_texts].</returns>
    Tensor<T> ComputeSimilarity(Tensor<T> imageEmbedding, Tensor<T> textEmbedding);

    /// <summary>
    /// Gets the dimensionality of the shared embedding space.
    /// </summary>
    int EmbeddingDim { get; }
}
```

#### File: `src/Interfaces/IContrastiveModel.cs`

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a contrastive vision-language model (like CLIP).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// Contrastive models learn by comparing:
///
/// **Positive pairs**: Matching image-text pairs (should be similar)
/// **Negative pairs**: Non-matching pairs (should be dissimilar)
///
/// Training process:
/// 1. Sample batch of (image, text) pairs
/// 2. Encode all images and texts
/// 3. Compute similarity matrix
/// 4. Maximize similarity of matching pairs
/// 5. Minimize similarity of non-matching pairs
///
/// This is called contrastive learning because it contrasts
/// positive and negative examples.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public interface IContrastiveModel<T> : IVisionLanguageModel<T>
{
    /// <summary>
    /// Computes contrastive loss for a batch of image-text pairs.
    /// </summary>
    /// <param name="images">Batch of images [batch, channels, height, width].</param>
    /// <param name="texts">Batch of text tokens [batch, sequence_length].</param>
    /// <returns>Contrastive loss value.</returns>
    T ComputeContrastiveLoss(Tensor<T> images, int[][] texts);

    /// <summary>
    /// Gets the temperature parameter for contrastive learning.
    /// Controls sharpness of similarity distribution.
    /// </summary>
    T Temperature { get; set; }

    /// <summary>
    /// Performs zero-shot image classification.
    /// </summary>
    /// <param name="image">Image to classify [channels, height, width].</param>
    /// <param name="classPrompts">Text prompts for each class.</param>
    /// <returns>Class probabilities [num_classes].</returns>
    Tensor<T> ZeroShotClassify(Tensor<T> image, string[] classPrompts);
}
```

#### File: `src/Interfaces/IVisualQuestionAnswering.cs`

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a model capable of visual question answering.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// Visual Question Answering (VQA) combines vision and language to
/// answer questions about images:
///
/// Example:
/// - Image: Photo of a red car
/// - Question: "What color is the car?"
/// - Answer: "Red"
///
/// The model must:
/// 1. Understand the image content
/// 2. Understand the question
/// 3. Reason about visual and linguistic information
/// 4. Generate or select the correct answer
///
/// VQA requires fine-grained alignment between vision and language,
/// often using cross-attention mechanisms.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public interface IVisualQuestionAnswering<T>
{
    /// <summary>
    /// Answers a question about an image.
    /// </summary>
    /// <param name="image">Input image [channels, height, width].</param>
    /// <param name="question">Question text as token IDs.</param>
    /// <param name="maxLength">Maximum length of generated answer.</param>
    /// <returns>Answer as token IDs.</returns>
    int[] AnswerQuestion(Tensor<T> image, int[] question, int maxLength = 20);

    /// <summary>
    /// Generates a caption for an image.
    /// </summary>
    /// <param name="image">Input image [channels, height, width].</param>
    /// <param name="maxLength">Maximum caption length.</param>
    /// <returns>Caption as token IDs.</returns>
    int[] GenerateCaption(Tensor<T> image, int maxLength = 50);

    /// <summary>
    /// Scores how well a caption matches an image.
    /// </summary>
    /// <param name="image">Input image.</param>
    /// <param name="caption">Caption text as token IDs.</param>
    /// <returns>Matching score (higher = better match).</returns>
    T ScoreImageTextMatch(Tensor<T> image, int[] caption);
}
```

### Step 2: Contrastive Loss Implementation

#### File: `src/VisionLanguage/Contrastive/ContrastiveLoss.cs`

```csharp
namespace AiDotNet.VisionLanguage.Contrastive;

using AiDotNet.Interfaces;
using AiDotNet.Mathematics;
using AiDotNet.Validation;

/// <summary>
/// Implements contrastive loss (InfoNCE) for vision-language models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// Contrastive loss encourages matching pairs to be similar and
/// non-matching pairs to be dissimilar.
///
/// Given a batch of N image-text pairs:
/// - Each image should be most similar to its corresponding text
/// - Each text should be most similar to its corresponding image
///
/// The loss is computed as cross-entropy over the similarity matrix:
///
/// For image i matching text i:
/// Loss_i = -log(exp(sim(I_i, T_i)/τ) / Σⱼ exp(sim(I_i, T_j)/τ))
///
/// Where τ (tau) is the temperature parameter controlling sharpness.
///
/// Symmetric loss:
/// Total = (Loss_image_to_text + Loss_text_to_image) / 2
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ContrastiveLoss<T>
{
    private readonly T _temperature;
    private readonly INumericOperations<T> _ops;

    /// <summary>
    /// Initializes a new instance of the <see cref="ContrastiveLoss{T}"/> class.
    /// </summary>
    /// <param name="temperature">Temperature parameter (typically 0.07).</param>
    /// <param name="ops">Numeric operations provider.</param>
    public ContrastiveLoss(T temperature, INumericOperations<T> ops)
    {
        Guard.NotNull(ops, nameof(ops));

        if (ops.LessThanOrEquals(temperature, ops.Zero))
        {
            throw new ArgumentException(
                "Temperature must be positive",
                nameof(temperature));
        }

        _temperature = temperature;
        _ops = ops;
    }

    /// <summary>
    /// Gets or sets the temperature parameter.
    /// </summary>
    public T Temperature => _temperature;

    /// <summary>
    /// Computes contrastive loss for a batch of embeddings.
    /// </summary>
    /// <param name="imageEmbeddings">Image embeddings [batch, dim].</param>
    /// <param name="textEmbeddings">Text embeddings [batch, dim].</param>
    /// <returns>Contrastive loss value.</returns>
    public T Compute(Tensor<T> imageEmbeddings, Tensor<T> textEmbeddings)
    {
        Guard.NotNull(imageEmbeddings, nameof(imageEmbeddings));
        Guard.NotNull(textEmbeddings, nameof(textEmbeddings));

        var imageShape = imageEmbeddings.Shape;
        var textShape = textEmbeddings.Shape;

        if (imageShape.Length != 2 || textShape.Length != 2)
        {
            throw new ArgumentException(
                "Embeddings must be 2D [batch, dim]");
        }

        if (imageShape[0] != textShape[0])
        {
            throw new ArgumentException(
                $"Batch sizes must match. Image: {imageShape[0]}, Text: {textShape[0]}");
        }

        if (imageShape[1] != textShape[1])
        {
            throw new ArgumentException(
                $"Embedding dimensions must match. Image: {imageShape[1]}, Text: {textShape[1]}");
        }

        int batchSize = imageShape[0];

        // Normalize embeddings to unit length
        var normalizedImages = NormalizeEmbeddings(imageEmbeddings);
        var normalizedTexts = NormalizeEmbeddings(textEmbeddings);

        // Compute similarity matrix: [batch, batch]
        var similarities = ComputeSimilarityMatrix(normalizedImages, normalizedTexts);

        // Scale by temperature
        similarities = ScaleByTemperature(similarities);

        // Compute image-to-text loss
        var i2tLoss = ComputeCrossEntropyLoss(similarities, batchSize);

        // Compute text-to-image loss (transpose similarities)
        var transposedSimilarities = TransposeSimilarities(similarities, batchSize);
        var t2iLoss = ComputeCrossEntropyLoss(transposedSimilarities, batchSize);

        // Average both directions
        var totalLoss = _ops.Add(i2tLoss, t2iLoss);
        return _ops.Divide(totalLoss, _ops.FromDouble(2.0));
    }

    private Tensor<T> NormalizeEmbeddings(Tensor<T> embeddings)
    {
        // L2 normalization: embedding / ||embedding||₂
        var shape = embeddings.Shape;
        int batch = shape[0];
        int dim = shape[1];

        var normalized = new Tensor<T>(shape);

        for (int b = 0; b < batch; b++)
        {
            // Compute L2 norm
            T sumSquares = _ops.Zero;
            for (int d = 0; d < dim; d++)
            {
                var val = embeddings[b, d];
                sumSquares = _ops.Add(sumSquares, _ops.Square(val));
            }
            var norm = _ops.Sqrt(sumSquares);

            // Avoid division by zero
            if (_ops.Equals(norm, _ops.Zero))
            {
                norm = _ops.FromDouble(1e-8);
            }

            // Normalize
            for (int d = 0; d < dim; d++)
            {
                normalized[b, d] = _ops.Divide(embeddings[b, d], norm);
            }
        }

        return normalized;
    }

    private Tensor<T> ComputeSimilarityMatrix(
        Tensor<T> imageEmbeddings,
        Tensor<T> textEmbeddings)
    {
        // Compute cosine similarity: images @ texts.T
        int batch = imageEmbeddings.Shape[0];
        int dim = imageEmbeddings.Shape[1];

        var similarities = new Tensor<T>(new[] { batch, batch });

        for (int i = 0; i < batch; i++)
        {
            for (int j = 0; j < batch; j++)
            {
                // Dot product (cosine similarity for normalized vectors)
                T dot = _ops.Zero;
                for (int d = 0; d < dim; d++)
                {
                    var prod = _ops.Multiply(imageEmbeddings[i, d], textEmbeddings[j, d]);
                    dot = _ops.Add(dot, prod);
                }
                similarities[i, j] = dot;
            }
        }

        return similarities;
    }

    private Tensor<T> ScaleByTemperature(Tensor<T> similarities)
    {
        var scaled = new Tensor<T>(similarities.Shape);

        for (int i = 0; i < similarities.Data.Length; i++)
        {
            scaled.Data[i] = _ops.Divide(similarities.Data[i], _temperature);
        }

        return scaled;
    }

    private T ComputeCrossEntropyLoss(Tensor<T> logits, int batchSize)
    {
        // Cross-entropy loss where target is the diagonal
        // Loss = -mean(log(softmax(logits)[i, i])) for i in batch

        T totalLoss = _ops.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            // Apply softmax to row i
            var softmax = ApplySoftmaxToRow(logits, i, batchSize);

            // Log probability of correct class (diagonal element)
            var prob = softmax[i];

            // Avoid log(0)
            if (_ops.LessThanOrEquals(prob, _ops.Zero))
            {
                prob = _ops.FromDouble(1e-8);
            }

            var logProb = _ops.Log(prob);
            totalLoss = _ops.Subtract(totalLoss, logProb); // Negative log likelihood
        }

        // Average over batch
        return _ops.Divide(totalLoss, _ops.FromDouble(batchSize));
    }

    private T[] ApplySoftmaxToRow(Tensor<T> logits, int row, int cols)
    {
        // Find max for numerical stability
        T maxLogit = logits[row, 0];
        for (int j = 1; j < cols; j++)
        {
            if (_ops.GreaterThan(logits[row, j], maxLogit))
            {
                maxLogit = logits[row, j];
            }
        }

        // Compute exp(x - max) and sum
        T sum = _ops.Zero;
        var exps = new T[cols];
        for (int j = 0; j < cols; j++)
        {
            var shifted = _ops.Subtract(logits[row, j], maxLogit);
            exps[j] = _ops.Exp(shifted);
            sum = _ops.Add(sum, exps[j]);
        }

        // Normalize
        var softmax = new T[cols];
        for (int j = 0; j < cols; j++)
        {
            softmax[j] = _ops.Divide(exps[j], sum);
        }

        return softmax;
    }

    private Tensor<T> TransposeSimilarities(Tensor<T> similarities, int size)
    {
        var transposed = new Tensor<T>(new[] { size, size });

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                transposed[i, j] = similarities[j, i];
            }
        }

        return transposed;
    }
}
```

### Step 3: Projection Head

#### File: `src/VisionLanguage/Projection/ProjectionHead.cs`

```csharp
namespace AiDotNet.VisionLanguage.Projection;

using AiDotNet.Interfaces;
using AiDotNet.Mathematics;
using AiDotNet.Validation;

/// <summary>
/// Implements a projection head to map encoders to shared embedding space.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// The projection head is a learnable linear layer that maps encoder outputs
/// to a shared embedding space where images and text can be compared.
///
/// Process:
/// 1. Vision encoder outputs features (e.g., 768-dim from ViT)
/// 2. Projection head maps to embedding space (e.g., 512-dim)
/// 3. Text encoder outputs features (e.g., 512-dim from Transformer)
/// 4. Projection head maps to same embedding space (512-dim)
/// 5. Now we can compute cosine similarity between image and text embeddings
///
/// The projection head is typically a single linear layer:
/// projection(x) = W·x + b
///
/// In CLIP, the projection is followed by L2 normalization.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ProjectionHead<T>
{
    private readonly Matrix<T> _weights;
    private readonly Vector<T> _bias;
    private readonly int _inputDim;
    private readonly int _outputDim;
    private readonly bool _normalize;

    /// <summary>
    /// Initializes a new instance of the <see cref="ProjectionHead{T}"/> class.
    /// </summary>
    /// <param name="inputDim">Input dimension.</param>
    /// <param name="outputDim">Output (embedding) dimension.</param>
    /// <param name="normalize">Whether to L2-normalize outputs.</param>
    /// <param name="ops">Numeric operations provider.</param>
    public ProjectionHead(
        int inputDim,
        int outputDim,
        bool normalize,
        INumericOperations<T> ops)
    {
        Guard.Positive(inputDim, nameof(inputDim));
        Guard.Positive(outputDim, nameof(outputDim));
        Guard.NotNull(ops, nameof(ops));

        _inputDim = inputDim;
        _outputDim = outputDim;
        _normalize = normalize;

        _weights = new Matrix<T>(outputDim, inputDim);
        _bias = new Vector<T>(outputDim);

        InitializeWeights(ops);
    }

    /// <summary>
    /// Gets the input dimension.
    /// </summary>
    public int InputDim => _inputDim;

    /// <summary>
    /// Gets the output dimension.
    /// </summary>
    public int OutputDim => _outputDim;

    /// <summary>
    /// Projects input features to embedding space.
    /// </summary>
    /// <param name="input">Input features [batch, input_dim].</param>
    /// <param name="ops">Numeric operations provider.</param>
    /// <returns>Projected embeddings [batch, output_dim].</returns>
    public Tensor<T> Forward(Tensor<T> input, INumericOperations<T> ops)
    {
        Guard.NotNull(input, nameof(input));
        Guard.NotNull(ops, nameof(ops));

        var shape = input.Shape;
        if (shape.Length != 2)
        {
            throw new ArgumentException(
                $"Input must be 2D [batch, input_dim], got: [{string.Join(", ", shape)}]",
                nameof(input));
        }

        int batch = shape[0];
        int inputDim = shape[1];

        if (inputDim != _inputDim)
        {
            throw new ArgumentException(
                $"Input dimension ({inputDim}) doesn't match expected ({_inputDim})",
                nameof(input));
        }

        // Linear projection: output = input @ weights.T + bias
        var output = new Tensor<T>(new[] { batch, _outputDim });

        for (int b = 0; b < batch; b++)
        {
            for (int o = 0; o < _outputDim; o++)
            {
                T sum = _bias[o];
                for (int i = 0; i < _inputDim; i++)
                {
                    var prod = ops.Multiply(input[b, i], _weights[o, i]);
                    sum = ops.Add(sum, prod);
                }
                output[b, o] = sum;
            }
        }

        // L2 normalization if requested
        if (_normalize)
        {
            output = NormalizeEmbeddings(output, ops);
        }

        return output;
    }

    private Tensor<T> NormalizeEmbeddings(Tensor<T> embeddings, INumericOperations<T> ops)
    {
        var shape = embeddings.Shape;
        int batch = shape[0];
        int dim = shape[1];

        var normalized = new Tensor<T>(shape);

        for (int b = 0; b < batch; b++)
        {
            // Compute L2 norm
            T sumSquares = ops.Zero;
            for (int d = 0; d < dim; d++)
            {
                var val = embeddings[b, d];
                sumSquares = ops.Add(sumSquares, ops.Square(val));
            }
            var norm = ops.Sqrt(sumSquares);

            // Avoid division by zero
            if (ops.Equals(norm, ops.Zero))
            {
                norm = ops.FromDouble(1e-8);
            }

            // Normalize
            for (int d = 0; d < dim; d++)
            {
                normalized[b, d] = ops.Divide(embeddings[b, d], norm);
            }
        }

        return normalized;
    }

    private void InitializeWeights(INumericOperations<T> ops)
    {
        // Xavier/Glorot initialization
        var random = new Random(42);
        double stddev = Math.Sqrt(2.0 / (_inputDim + _outputDim));

        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                double u1 = 1.0 - random.NextDouble();
                double u2 = 1.0 - random.NextDouble();
                double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                _weights[i, j] = ops.FromDouble(stddev * z0);
            }
        }

        // Bias initialized to zero
        for (int i = 0; i < _outputDim; i++)
        {
            _bias[i] = ops.Zero;
        }
    }
}
```

### Step 4: CLIP Model Implementation

#### File: `src/Models/VisionLanguage/CLIPModel.cs`

```csharp
namespace AiDotNet.Models.VisionLanguage;

using AiDotNet.Interfaces;
using AiDotNet.VisionLanguage.Projection;
using AiDotNet.VisionLanguage.Contrastive;
using AiDotNet.Mathematics;
using AiDotNet.Validation;

/// <summary>
/// Implements the CLIP (Contrastive Language-Image Pre-training) model.
/// </summary>
/// <remarks>
/// <para><b>Paper</b>: "Learning Transferable Visual Models From Natural Language Supervision"
/// by Radford et al. (OpenAI, 2021)
///
/// <b>Key Innovation</b>: Learn visual concepts from natural language supervision
/// at scale (400M image-text pairs).
///
/// <b>Architecture</b>:
/// ```
/// Image → Image Encoder → Image Projection → Image Embedding (512-dim)
///                                                  ↓
///                                          Cosine Similarity
///                                                  ↓
/// Text → Text Encoder → Text Projection → Text Embedding (512-dim)
/// ```
///
/// <b>Training Objective</b>: Contrastive learning
/// - Maximize similarity of matching (image, text) pairs
/// - Minimize similarity of non-matching pairs
///
/// <b>Zero-Shot Classification</b>:
/// Given an image and class names ["cat", "dog", "bird"]:
/// 1. Encode image
/// 2. Encode prompts: "a photo of a cat", "a photo of a dog", "a photo of a bird"
/// 3. Compute similarities
/// 4. Apply softmax to get probabilities
///
/// <b>Applications</b>:
/// - Zero-shot image classification
/// - Image-text retrieval
/// - Image search with natural language
/// - Transfer learning for vision tasks
///
/// <b>For Beginners</b>:
/// CLIP showed that pre-training on natural language descriptions is
/// more effective than traditional supervised learning on fixed classes.
/// It can classify images into arbitrary categories described in text,
/// without any task-specific training!
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CLIPModel<T> : IContrastiveModel<T>
{
    private readonly IVisionTransformer<T> _imageEncoder;
    private readonly ITransformer<T> _textEncoder;
    private readonly ProjectionHead<T> _imageProjection;
    private readonly ProjectionHead<T> _textProjection;
    private readonly ContrastiveLoss<T> _contrastiveLoss;
    private readonly INumericOperations<T> _ops;
    private readonly int _embeddingDim;
    private T _temperature;

    /// <summary>
    /// Initializes a new instance of the <see cref="CLIPModel{T}"/> class.
    /// </summary>
    /// <param name="imageEncoder">Vision encoder (e.g., ViT).</param>
    /// <param name="textEncoder">Text encoder (e.g., Transformer).</param>
    /// <param name="embeddingDim">Shared embedding dimension (e.g., 512).</param>
    /// <param name="temperature">Temperature for contrastive loss (e.g., 0.07).</param>
    /// <param name="ops">Numeric operations provider.</param>
    public CLIPModel(
        IVisionTransformer<T> imageEncoder,
        ITransformer<T> textEncoder,
        int embeddingDim,
        T temperature,
        INumericOperations<T> ops)
    {
        Guard.NotNull(imageEncoder, nameof(imageEncoder));
        Guard.NotNull(textEncoder, nameof(textEncoder));
        Guard.Positive(embeddingDim, nameof(embeddingDim));
        Guard.NotNull(ops, nameof(ops));

        _imageEncoder = imageEncoder;
        _textEncoder = textEncoder;
        _embeddingDim = embeddingDim;
        _temperature = temperature;
        _ops = ops;

        // Create projection heads
        _imageProjection = new ProjectionHead<T>(
            imageEncoder.EmbedDim,
            embeddingDim,
            normalize: true,
            ops);

        _textProjection = new ProjectionHead<T>(
            textEncoder.EmbedDim,
            embeddingDim,
            normalize: true,
            ops);

        // Create contrastive loss
        _contrastiveLoss = new ContrastiveLoss<T>(temperature, ops);
    }

    /// <inheritdoc/>
    public int EmbeddingDim => _embeddingDim;

    /// <inheritdoc/>
    public T Temperature
    {
        get => _temperature;
        set
        {
            if (_ops.LessThanOrEquals(value, _ops.Zero))
            {
                throw new ArgumentException("Temperature must be positive");
            }
            _temperature = value;
        }
    }

    /// <inheritdoc/>
    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        Guard.NotNull(image, nameof(image));

        // Extract features using vision encoder
        var features = _imageEncoder.Forward(image);

        // Project to shared embedding space
        var embeddings = _imageProjection.Forward(features, _ops);

        return embeddings;
    }

    /// <inheritdoc/>
    public Tensor<T> EncodeText(int[][] text)
    {
        Guard.NotNull(text, nameof(text));

        // Convert to tensor
        int batch = text.Length;
        int seqLen = text[0].Length;
        var textTensor = new Tensor<T>(new[] { batch, seqLen });

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                textTensor[b, s] = _ops.FromDouble(text[b][s]);
            }
        }

        // Extract features using text encoder
        var features = _textEncoder.Forward(textTensor);

        // Project to shared embedding space
        var embeddings = _textProjection.Forward(features, _ops);

        return embeddings;
    }

    /// <inheritdoc/>
    public Tensor<T> ComputeSimilarity(Tensor<T> imageEmbedding, Tensor<T> textEmbedding)
    {
        Guard.NotNull(imageEmbedding, nameof(imageEmbedding));
        Guard.NotNull(textEmbedding, nameof(textEmbedding));

        // Compute cosine similarity matrix
        int numImages = imageEmbedding.Shape[0];
        int numTexts = textEmbedding.Shape[0];
        int dim = imageEmbedding.Shape[1];

        var similarities = new Tensor<T>(new[] { numImages, numTexts });

        for (int i = 0; i < numImages; i++)
        {
            for (int j = 0; j < numTexts; j++)
            {
                // Dot product (embeddings are already normalized)
                T dot = _ops.Zero;
                for (int d = 0; d < dim; d++)
                {
                    var prod = _ops.Multiply(imageEmbedding[i, d], textEmbedding[j, d]);
                    dot = _ops.Add(dot, prod);
                }
                similarities[i, j] = dot;
            }
        }

        return similarities;
    }

    /// <inheritdoc/>
    public T ComputeContrastiveLoss(Tensor<T> images, int[][] texts)
    {
        Guard.NotNull(images, nameof(images));
        Guard.NotNull(texts, nameof(texts));

        // Encode images and texts
        var imageEmbeddings = EncodeImage(images);
        var textEmbeddings = EncodeText(texts);

        // Compute contrastive loss
        return _contrastiveLoss.Compute(imageEmbeddings, textEmbeddings);
    }

    /// <inheritdoc/>
    public Tensor<T> ZeroShotClassify(Tensor<T> image, string[] classPrompts)
    {
        Guard.NotNull(image, nameof(image));
        Guard.NotNull(classPrompts, nameof(classPrompts));

        // Add batch dimension if needed
        if (image.Shape.Length == 3)
        {
            var newShape = new int[4];
            newShape[0] = 1;
            Array.Copy(image.Shape, 0, newShape, 1, 3);
            image = image.Reshape(newShape);
        }

        // Encode image
        var imageEmbedding = EncodeImage(image);

        // Tokenize and encode class prompts
        var textTokens = TokenizePrompts(classPrompts);
        var textEmbeddings = EncodeText(textTokens);

        // Compute similarities
        var similarities = ComputeSimilarity(imageEmbedding, textEmbeddings);

        // Apply softmax to get probabilities
        var probabilities = ApplySoftmax(similarities);

        return probabilities;
    }

    private int[][] TokenizePrompts(string[] prompts)
    {
        // TODO: Implement proper tokenization using BPE tokenizer
        // For now, return placeholder
        var tokens = new int[prompts.Length][];
        for (int i = 0; i < prompts.Length; i++)
        {
            // Placeholder: would use actual tokenizer
            tokens[i] = new int[77]; // CLIP uses max length 77
        }
        return tokens;
    }

    private Tensor<T> ApplySoftmax(Tensor<T> logits)
    {
        // logits: [1, num_classes]
        int numClasses = logits.Shape[1];
        var probabilities = new Tensor<T>(logits.Shape);

        // Find max for numerical stability
        T maxLogit = logits[0, 0];
        for (int i = 1; i < numClasses; i++)
        {
            if (_ops.GreaterThan(logits[0, i], maxLogit))
            {
                maxLogit = logits[0, i];
            }
        }

        // Compute exp(x - max) and sum
        T sum = _ops.Zero;
        var exps = new T[numClasses];
        for (int i = 0; i < numClasses; i++)
        {
            var shifted = _ops.Subtract(logits[0, i], maxLogit);
            exps[i] = _ops.Exp(shifted);
            sum = _ops.Add(sum, exps[i]);
        }

        // Normalize
        for (int i = 0; i < numClasses; i++)
        {
            probabilities[0, i] = _ops.Divide(exps[i], sum);
        }

        return probabilities;
    }
}
```

---

## Testing Strategy

### Unit Tests

```csharp
namespace AiDotNetTests.UnitTests.VisionLanguage;

using AiDotNet.VisionLanguage.Contrastive;
using AiDotNet.Mathematics;
using Xunit;

public class ContrastiveLossTests
{
    [Fact]
    public void Compute_MatchingPairs_LowerLoss()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var loss = new ContrastiveLoss<double>(ops.FromDouble(0.07), ops);

        // Create embeddings where matching pairs are similar
        var imageEmbeds = new Tensor<double>(new[] { 4, 512 });
        var textEmbeds = new Tensor<double>(new[] { 4, 512 });

        // Fill with similar values for matching pairs
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 512; j++)
            {
                imageEmbeds[i, j] = i + 0.1 * j;
                textEmbeds[i, j] = i + 0.1 * j + 0.01; // Slight difference
            }
        }

        // Act
        var lossValue = loss.Compute(imageEmbeds, textEmbeds);

        // Assert
        Assert.True(lossValue >= 0); // Loss should be non-negative
        Assert.True(lossValue < 10); // Should be reasonably small for similar pairs
    }

    [Fact]
    public void Compute_NormalizedEmbeddings_ProducesSimilarities()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var loss = new ContrastiveLoss<double>(ops.FromDouble(0.07), ops);

        var imageEmbeds = new Tensor<double>(new[] { 2, 512 });
        var textEmbeds = new Tensor<double>(new[] { 2, 512 });

        // Fill with random values
        var random = new Random(42);
        for (int i = 0; i < imageEmbeds.Data.Length; i++)
        {
            imageEmbeds.Data[i] = random.NextDouble();
        }
        for (int i = 0; i < textEmbeds.Data.Length; i++)
        {
            textEmbeds.Data[i] = random.NextDouble();
        }

        // Act - Should not throw
        var lossValue = loss.Compute(imageEmbeds, textEmbeds);

        // Assert
        Assert.True(!double.IsNaN(lossValue));
        Assert.True(!double.IsInfinity(lossValue));
    }
}

public class ProjectionHeadTests
{
    [Fact]
    public void Forward_ProducesCorrectShape()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var projection = new ProjectionHead<double>(768, 512, normalize: true, ops);

        var input = new Tensor<double>(new[] { 4, 768 });

        // Act
        var output = projection.Forward(input, ops);

        // Assert
        Assert.Equal(new[] { 4, 512 }, output.Shape);
    }

    [Fact]
    public void Forward_WithNormalization_ProducesUnitVectors()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var projection = new ProjectionHead<double>(768, 512, normalize: true, ops);

        var input = new Tensor<double>(new[] { 2, 768 });

        // Fill with random values
        var random = new Random(42);
        for (int i = 0; i < input.Data.Length; i++)
        {
            input.Data[i] = random.NextDouble();
        }

        // Act
        var output = projection.Forward(input, ops);

        // Assert - Check that each embedding has unit length
        for (int b = 0; b < 2; b++)
        {
            double sumSquares = 0;
            for (int d = 0; d < 512; d++)
            {
                sumSquares += output[b, d] * output[b, d];
            }
            double norm = Math.Sqrt(sumSquares);
            Assert.True(Math.Abs(norm - 1.0) < 0.01); // Should be ~1.0
        }
    }
}

public class CLIPModelTests
{
    [Fact]
    public void ZeroShotClassify_ReturnsValidProbabilities()
    {
        // Arrange
        var ops = new DoubleNumericOperations();

        // Create mock encoders
        var imageEncoder = CreateMockVisionTransformer(ops);
        var textEncoder = CreateMockTextTransformer(ops);

        var model = new CLIPModel<double>(
            imageEncoder,
            textEncoder,
            embeddingDim: 512,
            temperature: ops.FromDouble(0.07),
            ops);

        var image = new Tensor<double>(new[] { 3, 224, 224 });
        var classes = new[] { "cat", "dog", "bird" };

        // Act
        var probabilities = model.ZeroShotClassify(image, classes);

        // Assert
        Assert.Equal(new[] { 1, 3 }, probabilities.Shape);

        // Probabilities should sum to ~1.0
        double sum = probabilities[0, 0] + probabilities[0, 1] + probabilities[0, 2];
        Assert.True(Math.Abs(sum - 1.0) < 0.01);

        // Each probability should be in [0, 1]
        for (int i = 0; i < 3; i++)
        {
            Assert.True(probabilities[0, i] >= 0);
            Assert.True(probabilities[0, i] <= 1.0);
        }
    }
}
```

---

## Training Strategy

### CLIP Pre-training

```csharp
/// <summary>
/// Pre-trains a CLIP model on image-text pairs.
/// </summary>
/// <remarks>
/// <b>Training Hyperparameters</b> (from CLIP paper):
///
/// 1. **Dataset**: 400M image-text pairs from internet
/// 2. **Batch Size**: 32,768 (distributed across many GPUs)
/// 3. **Optimizer**: Adam with β₁=0.9, β₂=0.98, ε=10⁻⁶
/// 4. **Learning Rate**: Cosine schedule, max LR = 5×10⁻⁴
/// 5. **Weight Decay**: 0.2
/// 6. **Warmup**: 2,000 steps
/// 7. **Training Steps**: 32 epochs (varies by model size)
/// 8. **Mixed Precision**: FP16 for efficiency
/// 9. **Temperature**: Learned, initialized to 0.07
///
/// <b>Data Augmentation</b>:
/// - Random crop and resize
/// - No other augmentation (natural language provides diversity)
///
/// <b>Training Time</b>:
/// - ViT-B/32: ~12 days on 256 V100 GPUs
/// - ViT-L/14: ~18 days on 256 V100 GPUs
///
/// <b>Evaluation</b>:
/// - Zero-shot accuracy on ImageNet
/// - Image-text retrieval on COCO, Flickr30K
/// - Robustness to distribution shift
/// </remarks>
public class CLIPTrainer<T>
{
    // TODO: Implement training with:
    // - Large-scale distributed training
    // - Mixed precision (FP16/BF16)
    // - Gradient accumulation
    // - Learning rate scheduling
    // - Checkpointing
    // - Zero-shot evaluation
}
```

### Expected Performance

| Model | Parameters | ImageNet Zero-Shot | ImageNet Fine-tuned |
|-------|-----------|-------------------|-------------------|
| CLIP ViT-B/32 | 150M | 63.2% | 85.3% |
| CLIP ViT-B/16 | 150M | 68.3% | 87.1% |
| CLIP ViT-L/14 | 427M | 75.5% | 88.3% |

---

## Common Pitfalls

### Pitfall 1: Not Normalizing Embeddings
**Problem**: Cosine similarity behaves poorly without normalization.
**Solution**: Always L2-normalize embeddings before computing similarity.

### Pitfall 2: Wrong Temperature
**Problem**: Temperature too high (soft) or too low (peaky) affects training.
**Solution**: Use 0.07 as default, consider learning it as a parameter.

### Pitfall 3: Batch Size Too Small
**Problem**: Contrastive learning needs many negative examples.
**Solution**: Use large batches (≥256) or accumulate gradients.

### Pitfall 4: Incorrect Loss Symmetry
**Problem**: Only computing image-to-text or text-to-image loss.
**Solution**: Average both directions for symmetric training.

### Pitfall 5: Poor Text Prompting
**Problem**: Using raw class names instead of proper prompts.
**Solution**: Use templates like "a photo of a {class}" for better zero-shot performance.

---

## Applications

### 1. Zero-Shot Image Classification
```csharp
var image = LoadImage("cat.jpg");
var classes = new[] { "cat", "dog", "bird", "fish" };
var probabilities = clipModel.ZeroShotClassify(image, classes);
// Output: [0.95, 0.03, 0.01, 0.01]
```

### 2. Image-Text Retrieval
```csharp
var images = LoadImages(1000); // 1000 images
var query = "a sunset over mountains";

var imageEmbeds = clipModel.EncodeImage(images);
var textEmbed = clipModel.EncodeText(new[] { Tokenize(query) });
var similarities = clipModel.ComputeSimilarity(imageEmbeds, textEmbed);

var topK = GetTopK(similarities, k: 10); // Top 10 matching images
```

### 3. Image Search Engine
```csharp
// Pre-compute image embeddings for database
var database = LoadImageDatabase(1_000_000);
var imageEmbeds = clipModel.EncodeImage(database);

// Real-time search
var userQuery = "puppies playing in grass";
var queryEmbed = clipModel.EncodeText(new[] { Tokenize(userQuery) });
var results = SearchTopK(imageEmbeds, queryEmbed, k: 20);
```

---

## Resources

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [BLIP Paper](https://arxiv.org/abs/2201.12086)
- [Flamingo Paper](https://arxiv.org/abs/2204.14198)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [CLIP Explained](https://github.com/openai/CLIP)
