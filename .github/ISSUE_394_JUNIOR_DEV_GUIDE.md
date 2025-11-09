# Junior Developer Implementation Guide: Issue #394
## Vision Transformers (ViT, Swin Transformer, DeiT)

### Overview
This guide will walk you through implementing Vision Transformers (ViT) and their advanced variants for AiDotNet. Vision Transformers revolutionized computer vision by applying the transformer architecture (originally from NLP) directly to images, proving that convolutions are not strictly necessary for excellent image understanding.

---

## Understanding Vision Transformers

### What Are Vision Transformers?

Traditional CNNs process images with convolutions (small sliding windows). Vision Transformers take a radically different approach:

1. **Split the image into patches**: Divide the image into fixed-size squares (e.g., 16x16 pixels)
2. **Flatten and embed patches**: Treat each patch as a "word" and convert it to a vector
3. **Add position information**: Tell the model where each patch came from
4. **Apply transformer layers**: Use self-attention to let patches "talk" to each other
5. **Make predictions**: Use the learned representations for classification or other tasks

**Real-World Analogy**:
- Traditional CNN: Like reading a book with a magnifying glass, examining small sections at a time
- Vision Transformer: Like reading a book by looking at all pages simultaneously and understanding how they relate to each other

### Key Concepts

#### 1. Patch Embedding
```
Input Image: 224x224x3 (Height x Width x Channels)
Patch Size: 16x16
Number of patches: (224/16) × (224/16) = 14 × 14 = 196 patches

Each patch: 16×16×3 = 768 values → Project to embedding dimension (e.g., 768)
Result: 196 vectors of size 768
```

**Why patches?**
- Transformers work best with sequences (like words in a sentence)
- Treating each pixel as a token would be computationally infeasible (224×224 = 50,176 tokens!)
- Patches are a sweet spot: small enough to capture details, large enough to be efficient

#### 2. Position Embeddings
```
Problem: Transformers have no built-in notion of spatial location
Solution: Add learnable position vectors to each patch embedding

patch_with_position = patch_embedding + position_embedding

Types:
- Learnable 1D: Simple learned vector for each position
- Learnable 2D: Separate embeddings for row and column
- Sinusoidal: Fixed mathematical patterns (like in original Transformer)
```

#### 3. Class Token (CLS)
```
Special learnable token prepended to the sequence:
[CLS] patch1 patch2 ... patch196

The CLS token:
- Learns to aggregate information from all patches
- Used for final classification (like a "summary" of the image)
- Inspired by BERT from NLP
```

#### 4. Multi-Head Self-Attention
```
Each patch attends to every other patch:

For patch i, compute:
- Query (Q): "What am I looking for?"
- Key (K): "What do I represent?"
- Value (V): "What information do I carry?"

Attention(Q,K,V) = softmax(QK^T / sqrt(d)) × V

This lets the model learn relationships like:
- "This patch is a cat's ear, related to the face patches"
- "These patches form the horizon line"
```

#### 5. Transformer Encoder Block
```
Standard transformer block:

Input → LayerNorm → Multi-Head Attention → Add (residual)
      ↓
      → LayerNorm → MLP (Feed-Forward) → Add (residual) → Output

MLP typically: Linear → GELU → Dropout → Linear → Dropout
```

### Vision Transformer Variants

#### ViT (Original)
- **Paper**: "An Image is Worth 16x16 Words" (Google, 2020)
- **Key Idea**: Pure transformer, no convolutions at all
- **Patch Size**: 16x16 or 32x32
- **Training**: Requires large datasets (ImageNet-21K, JFT-300M)
- **Transfer Learning**: Pre-train on huge datasets, fine-tune on smaller ones

**Architecture**:
```
Image (224×224×3)
  ↓
Patch Embedding (196 patches × 768 dim)
  ↓ Add CLS token
[CLS] + 196 patches
  ↓ Add position embeddings
Transformer Encoder × 12 layers
  ↓ Take CLS token output
MLP Head → Classification
```

#### DeiT (Data-efficient Image Transformer)
- **Paper**: "Training data-efficient image transformers" (Facebook, 2020)
- **Key Idea**: Train ViT on smaller datasets (ImageNet-1K) using:
  - **Distillation Token**: Learn from a CNN teacher (like ResNet)
  - **Knowledge Distillation**: Soft labels from teacher + hard labels
  - **Strong Data Augmentation**: RandAugment, Mixup, CutMix

**Architecture**:
```
Image (224×224×3)
  ↓
Patch Embedding
  ↓ Add CLS + Distillation tokens
[CLS] [DIST] patch1 ... patch196
  ↓ Add position embeddings
Transformer Encoder × 12 layers
  ↓
CLS output → classification loss
DIST output → distillation loss (match teacher)
```

**Why it matters**: Makes ViT practical without massive datasets

#### Swin Transformer
- **Paper**: "Swin Transformer: Hierarchical Vision Transformer" (Microsoft, 2021)
- **Key Idea**: Build a hierarchical transformer like CNNs
  - Start with small patches, merge them progressively
  - Use **shifted windows** for efficient attention
  - Create multi-scale features (like FPN in object detection)

**Shifted Window Attention**:
```
Problem: Full self-attention on all patches is O(N²) - too expensive
Solution:
- Partition image into non-overlapping windows (e.g., 7×7 patches)
- Compute attention only within each window: O(N)
- Shift windows by half the size every other layer
- This allows cross-window connections

Layer L:   [W1] [W2]     Layer L+1:  [Shifted W1] [Shifted W2]
           [W3] [W4]                  [Shifted W3] [Shifted W4]
```

**Hierarchical Architecture**:
```
Stage 1: H/4 × W/4 patches (small patches, many features)
  ↓ Patch Merging (merge 2×2 patches → reduce spatial dim, increase channels)
Stage 2: H/8 × W/8 patches
  ↓ Patch Merging
Stage 3: H/16 × W/16 patches
  ↓ Patch Merging
Stage 4: H/32 × W/32 patches
  ↓ Global Average Pooling → Classification

This creates multi-scale features useful for:
- Object detection
- Segmentation
- Dense prediction tasks
```

**Why it matters**:
- More efficient than ViT
- Better for downstream tasks (detection, segmentation)
- Works well on smaller datasets

---

## Architecture Overview

### File Structure
```
src/
├── Interfaces/
│   ├── IVisionTransformer.cs         # Base interface for all ViT variants
│   ├── IAttentionMechanism.cs        # Generic attention interface
│   └── IPatchEmbedding.cs            # Patch embedding interface
├── NeuralNetworks/
│   └── Transformers/
│       └── Vision/
│           ├── VisionTransformerBase.cs      # Base class
│           ├── ViTModel.cs                   # Original ViT
│           ├── DeiTModel.cs                  # DeiT with distillation
│           └── SwinTransformer.cs            # Swin Transformer
└── Transformers/
    ├── Embeddings/
    │   ├── PatchEmbedding.cs           # Convert image patches to embeddings
    │   ├── PositionEmbedding.cs        # Learnable position embeddings
    │   └── PositionEmbedding2D.cs      # 2D position embeddings
    ├── Attention/
    │   ├── MultiHeadSelfAttention.cs   # Standard self-attention
    │   └── WindowAttention.cs          # Swin's windowed attention
    └── Layers/
        ├── TransformerEncoderLayer.cs  # Standard transformer block
        ├── PatchMerging.cs             # Swin's patch merging
        └── DistillationHead.cs         # DeiT's distillation head
```

### Class Hierarchy
```
IVisionTransformer<T>
    ↓ implements IFullModel<T, Tensor<T>, Tensor<T>>
    ↓
VisionTransformerBase<T> (abstract)
    ├── ViTModel<T>           # Original Vision Transformer
    ├── DeiTModel<T>          # Data-efficient ViT
    └── SwinTransformer<T>    # Hierarchical ViT

IAttentionMechanism<T>
    ├── MultiHeadSelfAttention<T>   # Standard attention
    └── WindowAttention<T>           # Windowed attention

IPatchEmbedding<T>
    └── PatchEmbedding<T>     # Image → patch embeddings
```

---

## Step-by-Step Implementation

### Step 1: Core Interfaces

#### File: `src/Interfaces/IPatchEmbedding.cs`

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Represents patch embedding for Vision Transformers.
/// Converts image patches into embedding vectors.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// Patch embedding is like cutting a picture into small squares (patches)
/// and converting each square into a number sequence the transformer can understand.
///
/// Process:
/// 1. Split image into patches (e.g., 16×16 pixels)
/// 2. Flatten each patch into a vector
/// 3. Project to embedding dimension using a learned linear layer
///
/// Example with 224×224 image and 16×16 patches:
/// - Number of patches: (224/16) × (224/16) = 196
/// - Each patch: 16×16×3 = 768 raw values
/// - After embedding: 196 vectors of dimension D (e.g., 768)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public interface IPatchEmbedding<T>
{
    /// <summary>
    /// Converts an input image tensor into patch embeddings.
    /// </summary>
    /// <param name="input">
    /// Input image tensor with shape [batch, channels, height, width].
    /// Example: [32, 3, 224, 224] for a batch of 32 RGB images.
    /// </param>
    /// <param name="ops">Numeric operations provider.</param>
    /// <returns>
    /// Patch embeddings with shape [batch, num_patches, embed_dim].
    /// Example: [32, 196, 768] for 196 patches with 768-dimensional embeddings.
    /// </returns>
    Tensor<T> Embed(Tensor<T> input, INumericOperations<T> ops);

    /// <summary>
    /// Gets the number of patches produced from an image.
    /// </summary>
    /// <remarks>
    /// For a square image: num_patches = (image_size / patch_size)²
    /// For 224×224 image with 16×16 patches: 196 patches
    /// </remarks>
    int NumPatches { get; }

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    int EmbedDim { get; }

    /// <summary>
    /// Gets the patch size (height and width).
    /// </summary>
    int PatchSize { get; }
}
```

#### File: `src/Interfaces/IVisionTransformer.cs`

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a Vision Transformer model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// Vision Transformers apply the transformer architecture (from NLP) to images:
///
/// 1. Split image into patches
/// 2. Embed patches as tokens
/// 3. Add position information
/// 4. Process with transformer layers
/// 5. Make predictions from learned representations
///
/// Unlike CNNs that use convolutions, ViT uses self-attention to understand
/// relationships between different parts of the image.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public interface IVisionTransformer<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets the patch embedding layer.
    /// </summary>
    IPatchEmbedding<T> PatchEmbedding { get; }

    /// <summary>
    /// Gets the number of transformer layers.
    /// </summary>
    int NumLayers { get; }

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    int NumHeads { get; }

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    int EmbedDim { get; }

    /// <summary>
    /// Gets the MLP (feed-forward) hidden dimension.
    /// Typically 4 × EmbedDim.
    /// </summary>
    int MlpDim { get; }

    /// <summary>
    /// Extracts patch embeddings with positions for the input images.
    /// </summary>
    /// <param name="images">Input images [batch, channels, height, width].</param>
    /// <returns>Patch embeddings with positions [batch, num_patches+1, embed_dim].</returns>
    Tensor<T> ExtractPatches(Tensor<T> images);

    /// <summary>
    /// Applies transformer encoder layers to the embedded patches.
    /// </summary>
    /// <param name="embeddings">Patch embeddings [batch, num_patches, embed_dim].</param>
    /// <returns>Transformed embeddings [batch, num_patches, embed_dim].</returns>
    Tensor<T> ApplyTransformerLayers(Tensor<T> embeddings);
}
```

### Step 2: Patch Embedding Implementation

#### File: `src/Transformers/Embeddings/PatchEmbedding.cs`

```csharp
namespace AiDotNet.Transformers.Embeddings;

using AiDotNet.Interfaces;
using AiDotNet.Mathematics;
using AiDotNet.Validation;

/// <summary>
/// Implements patch embedding for Vision Transformers.
/// Converts 2D image patches into 1D embedding vectors.
/// </summary>
/// <remarks>
/// <para><b>Implementation Details:</b>
/// This is typically implemented as a convolutional layer with:
/// - Kernel size = patch size
/// - Stride = patch size (non-overlapping patches)
/// - Output channels = embedding dimension
///
/// This is equivalent to:
/// 1. Splitting image into patches
/// 2. Flattening each patch
/// 3. Applying a learned linear projection
///
/// But it's much more efficient in practice.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PatchEmbedding<T> : IPatchEmbedding<T>
{
    private readonly Matrix<T> _projectionWeights;
    private readonly Vector<T> _projectionBias;
    private readonly int _imageSize;
    private readonly int _patchSize;
    private readonly int _embedDim;
    private readonly int _numPatches;

    /// <summary>
    /// Initializes a new instance of the <see cref="PatchEmbedding{T}"/> class.
    /// </summary>
    /// <param name="imageSize">Size of the input image (assumed square).</param>
    /// <param name="patchSize">Size of each patch (assumed square).</param>
    /// <param name="inChannels">Number of input channels (e.g., 3 for RGB).</param>
    /// <param name="embedDim">Dimension of the embedding vectors.</param>
    /// <param name="ops">Numeric operations provider.</param>
    public PatchEmbedding(
        int imageSize,
        int patchSize,
        int inChannels,
        int embedDim,
        INumericOperations<T> ops)
    {
        Guard.Positive(imageSize, nameof(imageSize));
        Guard.Positive(patchSize, nameof(patchSize));
        Guard.Positive(inChannels, nameof(inChannels));
        Guard.Positive(embedDim, nameof(embedDim));
        Guard.NotNull(ops, nameof(ops));

        if (imageSize % patchSize != 0)
        {
            throw new ArgumentException(
                $"Image size ({imageSize}) must be divisible by patch size ({patchSize}).",
                nameof(imageSize));
        }

        _imageSize = imageSize;
        _patchSize = patchSize;
        _embedDim = embedDim;
        _numPatches = (imageSize / patchSize) * (imageSize / patchSize);

        // Initialize projection as a "convolution" with kernel_size = stride = patch_size
        int patchValues = patchSize * patchSize * inChannels;
        _projectionWeights = new Matrix<T>(embedDim, patchValues);
        _projectionBias = new Vector<T>(embedDim);

        // Xavier/Glorot initialization
        InitializeWeights(ops, patchValues);
    }

    /// <inheritdoc/>
    public int NumPatches => _numPatches;

    /// <inheritdoc/>
    public int EmbedDim => _embedDim;

    /// <inheritdoc/>
    public int PatchSize => _patchSize;

    /// <inheritdoc/>
    public Tensor<T> Embed(Tensor<T> input, INumericOperations<T> ops)
    {
        Guard.NotNull(input, nameof(input));
        Guard.NotNull(ops, nameof(ops));

        // Input shape: [batch, channels, height, width]
        var shape = input.Shape;
        if (shape.Length != 4)
        {
            throw new ArgumentException(
                $"Expected 4D input tensor [batch, channels, height, width], got shape: [{string.Join(", ", shape)}]",
                nameof(input));
        }

        int batch = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];

        if (height != _imageSize || width != _imageSize)
        {
            throw new ArgumentException(
                $"Expected image size {_imageSize}×{_imageSize}, got {height}×{width}",
                nameof(input));
        }

        // Extract patches
        var patches = ExtractPatches(input, batch, channels, height, width, ops);

        // Project patches to embedding dimension
        // patches shape: [batch * num_patches, patch_values]
        // weights shape: [embed_dim, patch_values]
        // result shape: [batch * num_patches, embed_dim]
        var embedded = patches.MatrixMultiply(_projectionWeights.Transpose(), ops);

        // Add bias
        embedded = AddBias(embedded, ops);

        // Reshape to [batch, num_patches, embed_dim]
        var outputShape = new[] { batch, _numPatches, _embedDim };
        return embedded.Reshape(outputShape);
    }

    private Tensor<T> ExtractPatches(
        Tensor<T> input,
        int batch,
        int channels,
        int height,
        int width,
        INumericOperations<T> ops)
    {
        int patchesPerRow = height / _patchSize;
        int patchesPerCol = width / _patchSize;
        int patchValues = _patchSize * _patchSize * channels;

        var patches = new Tensor<T>(new[] { batch * _numPatches, patchValues });

        for (int b = 0; b < batch; b++)
        {
            int patchIdx = 0;
            for (int i = 0; i < patchesPerRow; i++)
            {
                for (int j = 0; j < patchesPerCol; j++)
                {
                    int startH = i * _patchSize;
                    int startW = j * _patchSize;

                    // Extract and flatten patch
                    int flatIdx = 0;
                    for (int c = 0; c < channels; c++)
                    {
                        for (int ph = 0; ph < _patchSize; ph++)
                        {
                            for (int pw = 0; pw < _patchSize; pw++)
                            {
                                var value = input[b, c, startH + ph, startW + pw];
                                patches[b * _numPatches + patchIdx, flatIdx] = value;
                                flatIdx++;
                            }
                        }
                    }
                    patchIdx++;
                }
            }
        }

        return patches;
    }

    private Tensor<T> AddBias(Tensor<T> embedded, INumericOperations<T> ops)
    {
        // Broadcast bias across all positions
        var shape = embedded.Shape;
        var result = new Tensor<T>(shape);

        for (int i = 0; i < shape[0]; i++)
        {
            for (int j = 0; j < shape[1]; j++)
            {
                result[i, j] = ops.Add(embedded[i, j], _projectionBias[j]);
            }
        }

        return result;
    }

    private void InitializeWeights(INumericOperations<T> ops, int fanIn)
    {
        // Xavier/Glorot initialization: stddev = sqrt(2 / (fan_in + fan_out))
        double stddev = Math.Sqrt(2.0 / (fanIn + _embedDim));
        var random = new Random(42);

        for (int i = 0; i < _projectionWeights.Rows; i++)
        {
            for (int j = 0; j < _projectionWeights.Columns; j++)
            {
                double value = SampleGaussian(random, 0, stddev);
                _projectionWeights[i, j] = ops.FromDouble(value);
            }
        }

        // Bias initialized to zero
        for (int i = 0; i < _projectionBias.Length; i++)
        {
            _projectionBias[i] = ops.Zero;
        }
    }

    private static double SampleGaussian(Random random, double mean, double stddev)
    {
        // Box-Muller transform
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return mean + stddev * z0;
    }
}
```

### Step 3: Position Embeddings

#### File: `src/Transformers/Embeddings/PositionEmbedding.cs`

```csharp
namespace AiDotNet.Transformers.Embeddings;

using AiDotNet.Interfaces;
using AiDotNet.Mathematics;
using AiDotNet.Validation;

/// <summary>
/// Implements learnable position embeddings for transformers.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// Position embeddings tell the model where each patch is located in the image.
///
/// Since transformers process all patches simultaneously (unlike CNNs that preserve
/// spatial structure), we need to explicitly add position information.
///
/// Types:
/// - **Learnable**: Start with random values, learn during training
/// - **Sinusoidal**: Fixed mathematical pattern (like in original Transformer)
///
/// This implementation uses learnable embeddings, which work better for vision
/// tasks in practice.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PositionEmbedding<T>
{
    private readonly Matrix<T> _embeddings;
    private readonly int _numPositions;
    private readonly int _embedDim;

    /// <summary>
    /// Initializes a new instance of the <see cref="PositionEmbedding{T}"/> class.
    /// </summary>
    /// <param name="numPositions">Number of positions (num_patches + 1 for CLS token).</param>
    /// <param name="embedDim">Dimension of the embeddings.</param>
    /// <param name="ops">Numeric operations provider.</param>
    public PositionEmbedding(int numPositions, int embedDim, INumericOperations<T> ops)
    {
        Guard.Positive(numPositions, nameof(numPositions));
        Guard.Positive(embedDim, nameof(embedDim));
        Guard.NotNull(ops, nameof(ops));

        _numPositions = numPositions;
        _embedDim = embedDim;
        _embeddings = new Matrix<T>(numPositions, embedDim);

        InitializeEmbeddings(ops);
    }

    /// <summary>
    /// Gets the number of positions.
    /// </summary>
    public int NumPositions => _numPositions;

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbedDim => _embedDim;

    /// <summary>
    /// Adds position embeddings to the input tensor.
    /// </summary>
    /// <param name="input">Input tensor with shape [batch, seq_len, embed_dim].</param>
    /// <param name="ops">Numeric operations provider.</param>
    /// <returns>Input with position embeddings added.</returns>
    public Tensor<T> AddPositionEmbeddings(Tensor<T> input, INumericOperations<T> ops)
    {
        Guard.NotNull(input, nameof(input));
        Guard.NotNull(ops, nameof(ops));

        var shape = input.Shape;
        if (shape.Length != 3)
        {
            throw new ArgumentException(
                $"Expected 3D input [batch, seq_len, embed_dim], got shape: [{string.Join(", ", shape)}]",
                nameof(input));
        }

        int batch = shape[0];
        int seqLen = shape[1];
        int embedDim = shape[2];

        if (seqLen > _numPositions)
        {
            throw new ArgumentException(
                $"Sequence length ({seqLen}) exceeds maximum positions ({_numPositions})",
                nameof(input));
        }

        if (embedDim != _embedDim)
        {
            throw new ArgumentException(
                $"Embedding dimension ({embedDim}) doesn't match expected ({_embedDim})",
                nameof(input));
        }

        var result = new Tensor<T>(shape);

        // Add position embeddings: result[b, i, j] = input[b, i, j] + embeddings[i, j]
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < embedDim; j++)
                {
                    result[b, i, j] = ops.Add(input[b, i, j], _embeddings[i, j]);
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Gets the position embedding for a specific position.
    /// </summary>
    /// <param name="position">The position index.</param>
    /// <returns>The position embedding vector.</returns>
    public Vector<T> GetPositionEmbedding(int position)
    {
        Guard.InRange(position, 0, _numPositions - 1, nameof(position));

        var embedding = new Vector<T>(_embedDim);
        for (int i = 0; i < _embedDim; i++)
        {
            embedding[i] = _embeddings[position, i];
        }

        return embedding;
    }

    private void InitializeEmbeddings(INumericOperations<T> ops)
    {
        // Initialize with small random values (truncated normal distribution)
        var random = new Random(42);
        double stddev = 0.02; // Standard deviation commonly used in ViT

        for (int i = 0; i < _numPositions; i++)
        {
            for (int j = 0; j < _embedDim; j++)
            {
                double value = SampleTruncatedNormal(random, 0, stddev, -2 * stddev, 2 * stddev);
                _embeddings[i, j] = ops.FromDouble(value);
            }
        }
    }

    private static double SampleTruncatedNormal(
        Random random,
        double mean,
        double stddev,
        double minValue,
        double maxValue)
    {
        // Sample from truncated normal distribution
        double value;
        do
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            value = mean + stddev * z0;
        }
        while (value < minValue || value > maxValue);

        return value;
    }
}
```

### Step 4: Multi-Head Self-Attention

#### File: `src/Transformers/Attention/MultiHeadSelfAttention.cs`

```csharp
namespace AiDotNet.Transformers.Attention;

using AiDotNet.Interfaces;
using AiDotNet.Mathematics;
using AiDotNet.Validation;

/// <summary>
/// Implements multi-head self-attention mechanism.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// Self-attention lets each patch "look at" all other patches to understand context.
///
/// Process for each patch:
/// 1. Create Query (Q): "What am I looking for?"
/// 2. Create Key (K) for all patches: "What do I represent?"
/// 3. Create Value (V) for all patches: "What information do I have?"
/// 4. Compute attention weights: softmax(Q·K^T / sqrt(d))
/// 5. Aggregate information: weighted sum of Values
///
/// Multi-head means doing this multiple times in parallel with different
/// learned transformations, then combining the results.
///
/// Example: 8 heads with 768-dim embeddings
/// - Each head works with 768/8 = 96 dimensions
/// - Each head can learn to focus on different aspects (edges, textures, objects)
/// - Results are concatenated and projected back to 768 dims
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MultiHeadSelfAttention<T> : IAttentionMechanism<T>
{
    private readonly int _embedDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly double _scale;

    // Projection matrices
    private readonly Matrix<T> _queryWeights;
    private readonly Matrix<T> _keyWeights;
    private readonly Matrix<T> _valueWeights;
    private readonly Matrix<T> _outputWeights;

    private readonly Vector<T> _queryBias;
    private readonly Vector<T> _keyBias;
    private readonly Vector<T> _valueBias;
    private readonly Vector<T> _outputBias;

    /// <summary>
    /// Initializes a new instance of the <see cref="MultiHeadSelfAttention{T}"/> class.
    /// </summary>
    /// <param name="embedDim">Embedding dimension.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="ops">Numeric operations provider.</param>
    public MultiHeadSelfAttention(int embedDim, int numHeads, INumericOperations<T> ops)
    {
        Guard.Positive(embedDim, nameof(embedDim));
        Guard.Positive(numHeads, nameof(numHeads));
        Guard.NotNull(ops, nameof(ops));

        if (embedDim % numHeads != 0)
        {
            throw new ArgumentException(
                $"Embedding dimension ({embedDim}) must be divisible by number of heads ({numHeads})",
                nameof(embedDim));
        }

        _embedDim = embedDim;
        _numHeads = numHeads;
        _headDim = embedDim / numHeads;
        _scale = 1.0 / Math.Sqrt(_headDim);

        // Initialize projection matrices
        _queryWeights = new Matrix<T>(embedDim, embedDim);
        _keyWeights = new Matrix<T>(embedDim, embedDim);
        _valueWeights = new Matrix<T>(embedDim, embedDim);
        _outputWeights = new Matrix<T>(embedDim, embedDim);

        _queryBias = new Vector<T>(embedDim);
        _keyBias = new Vector<T>(embedDim);
        _valueBias = new Vector<T>(embedDim);
        _outputBias = new Vector<T>(embedDim);

        InitializeWeights(ops);
    }

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbedDim => _embedDim;

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <inheritdoc/>
    public Tensor<T> Forward(Tensor<T> input, INumericOperations<T> ops)
    {
        Guard.NotNull(input, nameof(input));
        Guard.NotNull(ops, nameof(ops));

        // Input shape: [batch, seq_len, embed_dim]
        var shape = input.Shape;
        if (shape.Length != 3)
        {
            throw new ArgumentException(
                $"Expected 3D input [batch, seq_len, embed_dim], got shape: [{string.Join(", ", shape)}]",
                nameof(input));
        }

        int batch = shape[0];
        int seqLen = shape[1];
        int embedDim = shape[2];

        if (embedDim != _embedDim)
        {
            throw new ArgumentException(
                $"Input embedding dimension ({embedDim}) doesn't match expected ({_embedDim})",
                nameof(input));
        }

        // Project to Q, K, V
        var queries = ProjectAndReshape(input, _queryWeights, _queryBias, batch, seqLen, ops);
        var keys = ProjectAndReshape(input, _keyWeights, _keyBias, batch, seqLen, ops);
        var values = ProjectAndReshape(input, _valueWeights, _valueBias, batch, seqLen, ops);

        // Compute attention
        // queries, keys, values shape: [batch * num_heads, seq_len, head_dim]
        var attended = ComputeAttention(queries, keys, values, batch, seqLen, ops);

        // Reshape back: [batch * num_heads, seq_len, head_dim] → [batch, seq_len, embed_dim]
        var concatenated = ConcatenateHeads(attended, batch, seqLen);

        // Output projection
        var output = ApplyOutputProjection(concatenated, ops);

        return output;
    }

    private Tensor<T> ProjectAndReshape(
        Tensor<T> input,
        Matrix<T> weights,
        Vector<T> bias,
        int batch,
        int seqLen,
        INumericOperations<T> ops)
    {
        // Flatten batch and seq_len: [batch, seq_len, embed_dim] → [batch * seq_len, embed_dim]
        var flattened = input.Reshape(new[] { batch * seqLen, _embedDim });

        // Project: [batch * seq_len, embed_dim] × [embed_dim, embed_dim] → [batch * seq_len, embed_dim]
        var projected = flattened.MatrixMultiply(weights.Transpose(), ops);

        // Add bias
        projected = AddBias(projected, bias, ops);

        // Reshape to [batch, seq_len, num_heads, head_dim]
        var reshaped = projected.Reshape(new[] { batch, seqLen, _numHeads, _headDim });

        // Transpose to [batch, num_heads, seq_len, head_dim]
        var transposed = reshaped.Transpose(0, 2, 1, 3);

        // Merge batch and heads: [batch * num_heads, seq_len, head_dim]
        return transposed.Reshape(new[] { batch * _numHeads, seqLen, _headDim });
    }

    private Tensor<T> ComputeAttention(
        Tensor<T> queries,
        Tensor<T> keys,
        Tensor<T> values,
        int batch,
        int seqLen,
        INumericOperations<T> ops)
    {
        // queries, keys, values: [batch * num_heads, seq_len, head_dim]

        // Compute attention scores: Q × K^T / sqrt(head_dim)
        // [batch * num_heads, seq_len, head_dim] × [batch * num_heads, head_dim, seq_len]
        // → [batch * num_heads, seq_len, seq_len]
        var scores = new Tensor<T>(new[] { batch * _numHeads, seqLen, seqLen });

        for (int b = 0; b < batch * _numHeads; b++)
        {
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < seqLen; j++)
                {
                    T score = ops.Zero;
                    for (int k = 0; k < _headDim; k++)
                    {
                        var q = queries[b, i, k];
                        var key = keys[b, j, k];
                        score = ops.Add(score, ops.Multiply(q, key));
                    }
                    // Scale by 1/sqrt(head_dim)
                    scores[b, i, j] = ops.Multiply(score, ops.FromDouble(_scale));
                }
            }
        }

        // Apply softmax to get attention weights
        var attentionWeights = ApplySoftmax(scores, seqLen, ops);

        // Apply attention weights to values
        // [batch * num_heads, seq_len, seq_len] × [batch * num_heads, seq_len, head_dim]
        // → [batch * num_heads, seq_len, head_dim]
        var attended = new Tensor<T>(new[] { batch * _numHeads, seqLen, _headDim });

        for (int b = 0; b < batch * _numHeads; b++)
        {
            for (int i = 0; i < seqLen; i++)
            {
                for (int k = 0; k < _headDim; k++)
                {
                    T sum = ops.Zero;
                    for (int j = 0; j < seqLen; j++)
                    {
                        var weight = attentionWeights[b, i, j];
                        var value = values[b, j, k];
                        sum = ops.Add(sum, ops.Multiply(weight, value));
                    }
                    attended[b, i, k] = sum;
                }
            }
        }

        return attended;
    }

    private Tensor<T> ApplySoftmax(Tensor<T> scores, int seqLen, INumericOperations<T> ops)
    {
        var result = new Tensor<T>(scores.Shape);
        int batchHeads = scores.Shape[0];

        for (int b = 0; b < batchHeads; b++)
        {
            for (int i = 0; i < seqLen; i++)
            {
                // Find max for numerical stability
                T maxScore = scores[b, i, 0];
                for (int j = 1; j < seqLen; j++)
                {
                    if (ops.GreaterThan(scores[b, i, j], maxScore))
                    {
                        maxScore = scores[b, i, j];
                    }
                }

                // Compute exp(x - max) and sum
                T sum = ops.Zero;
                var exps = new T[seqLen];
                for (int j = 0; j < seqLen; j++)
                {
                    var shifted = ops.Subtract(scores[b, i, j], maxScore);
                    exps[j] = ops.Exp(shifted);
                    sum = ops.Add(sum, exps[j]);
                }

                // Normalize
                for (int j = 0; j < seqLen; j++)
                {
                    result[b, i, j] = ops.Divide(exps[j], sum);
                }
            }
        }

        return result;
    }

    private Tensor<T> ConcatenateHeads(Tensor<T> attended, int batch, int seqLen)
    {
        // attended: [batch * num_heads, seq_len, head_dim]
        // output: [batch, seq_len, embed_dim]
        var output = new Tensor<T>(new[] { batch, seqLen, _embedDim });

        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < seqLen; i++)
            {
                int idx = 0;
                for (int h = 0; h < _numHeads; h++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        output[b, i, idx] = attended[b * _numHeads + h, i, d];
                        idx++;
                    }
                }
            }
        }

        return output;
    }

    private Tensor<T> ApplyOutputProjection(Tensor<T> input, INumericOperations<T> ops)
    {
        var shape = input.Shape;
        int batch = shape[0];
        int seqLen = shape[1];

        // Flatten: [batch, seq_len, embed_dim] → [batch * seq_len, embed_dim]
        var flattened = input.Reshape(new[] { batch * seqLen, _embedDim });

        // Project
        var projected = flattened.MatrixMultiply(_outputWeights.Transpose(), ops);

        // Add bias
        projected = AddBias(projected, _outputBias, ops);

        // Reshape back
        return projected.Reshape(new[] { batch, seqLen, _embedDim });
    }

    private Tensor<T> AddBias(Tensor<T> input, Vector<T> bias, INumericOperations<T> ops)
    {
        var shape = input.Shape;
        var result = new Tensor<T>(shape);

        for (int i = 0; i < shape[0]; i++)
        {
            for (int j = 0; j < shape[1]; j++)
            {
                result[i, j] = ops.Add(input[i, j], bias[j]);
            }
        }

        return result;
    }

    private void InitializeWeights(INumericOperations<T> ops)
    {
        // Xavier/Glorot initialization
        double stddev = Math.Sqrt(2.0 / (_embedDim + _embedDim));
        var random = new Random(42);

        InitializeMatrix(_queryWeights, random, stddev, ops);
        InitializeMatrix(_keyWeights, random, stddev, ops);
        InitializeMatrix(_valueWeights, random, stddev, ops);
        InitializeMatrix(_outputWeights, random, stddev, ops);

        // Biases initialized to zero
        InitializeVectorToZero(_queryBias, ops);
        InitializeVectorToZero(_keyBias, ops);
        InitializeVectorToZero(_valueBias, ops);
        InitializeVectorToZero(_outputBias, ops);
    }

    private void InitializeMatrix(Matrix<T> matrix, Random random, double stddev, INumericOperations<T> ops)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                double value = SampleGaussian(random, 0, stddev);
                matrix[i, j] = ops.FromDouble(value);
            }
        }
    }

    private void InitializeVectorToZero(Vector<T> vector, INumericOperations<T> ops)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = ops.Zero;
        }
    }

    private static double SampleGaussian(Random random, double mean, double stddev)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return mean + stddev * z0;
    }
}
```

### Step 5: Transformer Encoder Layer

#### File: `src/Transformers/Layers/TransformerEncoderLayer.cs`

```csharp
namespace AiDotNet.Transformers.Layers;

using AiDotNet.Interfaces;
using AiDotNet.Transformers.Attention;
using AiDotNet.Mathematics;
using AiDotNet.Validation;

/// <summary>
/// Implements a standard transformer encoder layer.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// A transformer encoder layer consists of two main components:
///
/// 1. **Multi-Head Self-Attention**: Let patches communicate with each other
/// 2. **Feed-Forward Network (MLP)**: Process each patch independently
///
/// Both components use:
/// - **LayerNorm**: Normalize activations for stable training
/// - **Residual Connections**: Add input to output (helps gradients flow)
/// - **Dropout**: Randomly zero some values during training (regularization)
///
/// Structure:
/// ```
/// Input
///   ↓
///   LayerNorm → Multi-Head Attention → Dropout → Add (residual) → Output1
///   ↓
///   LayerNorm → MLP (Linear → GELU → Dropout → Linear → Dropout) → Add (residual) → Output2
/// ```
///
/// This design, called "Pre-LN Transformer", is more stable for training than
/// the original "Post-LN" design.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TransformerEncoderLayer<T>
{
    private readonly MultiHeadSelfAttention<T> _attention;
    private readonly int _embedDim;
    private readonly int _mlpDim;
    private readonly double _dropoutRate;

    // Layer normalization parameters
    private readonly Vector<T> _norm1Gamma;
    private readonly Vector<T> _norm1Beta;
    private readonly Vector<T> _norm2Gamma;
    private readonly Vector<T> _norm2Beta;

    // MLP parameters
    private readonly Matrix<T> _mlp1Weights;
    private readonly Vector<T> _mlp1Bias;
    private readonly Matrix<T> _mlp2Weights;
    private readonly Vector<T> _mlp2Bias;

    /// <summary>
    /// Initializes a new instance of the <see cref="TransformerEncoderLayer{T}"/> class.
    /// </summary>
    /// <param name="embedDim">Embedding dimension.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="mlpDim">MLP hidden dimension (typically 4 × embed_dim).</param>
    /// <param name="dropoutRate">Dropout rate for regularization.</param>
    /// <param name="ops">Numeric operations provider.</param>
    public TransformerEncoderLayer(
        int embedDim,
        int numHeads,
        int mlpDim,
        double dropoutRate,
        INumericOperations<T> ops)
    {
        Guard.Positive(embedDim, nameof(embedDim));
        Guard.Positive(numHeads, nameof(numHeads));
        Guard.Positive(mlpDim, nameof(mlpDim));
        Guard.InRange(dropoutRate, 0.0, 1.0, nameof(dropoutRate));
        Guard.NotNull(ops, nameof(ops));

        _embedDim = embedDim;
        _mlpDim = mlpDim;
        _dropoutRate = dropoutRate;

        _attention = new MultiHeadSelfAttention<T>(embedDim, numHeads, ops);

        // Initialize layer norm parameters
        _norm1Gamma = new Vector<T>(embedDim);
        _norm1Beta = new Vector<T>(embedDim);
        _norm2Gamma = new Vector<T>(embedDim);
        _norm2Beta = new Vector<T>(embedDim);

        // Initialize MLP parameters
        _mlp1Weights = new Matrix<T>(mlpDim, embedDim);
        _mlp1Bias = new Vector<T>(mlpDim);
        _mlp2Weights = new Matrix<T>(embedDim, mlpDim);
        _mlp2Bias = new Vector<T>(embedDim);

        InitializeParameters(ops);
    }

    /// <summary>
    /// Forward pass through the transformer encoder layer.
    /// </summary>
    /// <param name="input">Input tensor [batch, seq_len, embed_dim].</param>
    /// <param name="ops">Numeric operations provider.</param>
    /// <param name="training">Whether in training mode (applies dropout).</param>
    /// <returns>Output tensor [batch, seq_len, embed_dim].</returns>
    public Tensor<T> Forward(Tensor<T> input, INumericOperations<T> ops, bool training = false)
    {
        Guard.NotNull(input, nameof(input));
        Guard.NotNull(ops, nameof(ops));

        // Attention block: LayerNorm → Attention → Dropout → Residual
        var normalized1 = ApplyLayerNorm(input, _norm1Gamma, _norm1Beta, ops);
        var attended = _attention.Forward(normalized1, ops);

        if (training)
        {
            attended = ApplyDropout(attended, _dropoutRate, ops);
        }

        var afterAttention = AddResidual(input, attended, ops);

        // MLP block: LayerNorm → MLP → Dropout → Residual
        var normalized2 = ApplyLayerNorm(afterAttention, _norm2Gamma, _norm2Beta, ops);
        var mlpOutput = ApplyMLP(normalized2, ops, training);

        if (training)
        {
            mlpOutput = ApplyDropout(mlpOutput, _dropoutRate, ops);
        }

        var output = AddResidual(afterAttention, mlpOutput, ops);

        return output;
    }

    private Tensor<T> ApplyLayerNorm(
        Tensor<T> input,
        Vector<T> gamma,
        Vector<T> beta,
        INumericOperations<T> ops)
    {
        // Layer normalization over the last dimension
        var shape = input.Shape;
        int batch = shape[0];
        int seqLen = shape[1];
        int embedDim = shape[2];

        var output = new Tensor<T>(shape);
        double epsilon = 1e-5;

        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < seqLen; i++)
            {
                // Compute mean
                T sum = ops.Zero;
                for (int j = 0; j < embedDim; j++)
                {
                    sum = ops.Add(sum, input[b, i, j]);
                }
                var meanVal = ops.Divide(sum, ops.FromDouble(embedDim));

                // Compute variance
                T sumSq = ops.Zero;
                for (int j = 0; j < embedDim; j++)
                {
                    var diff = ops.Subtract(input[b, i, j], meanVal);
                    sumSq = ops.Add(sumSq, ops.Square(diff));
                }
                var variance = ops.Divide(sumSq, ops.FromDouble(embedDim));

                // Normalize
                var stdDev = ops.Sqrt(ops.Add(variance, ops.FromDouble(epsilon)));
                for (int j = 0; j < embedDim; j++)
                {
                    var normalized = ops.Divide(
                        ops.Subtract(input[b, i, j], meanVal),
                        stdDev);

                    // Apply affine transformation
                    output[b, i, j] = ops.Add(
                        ops.Multiply(normalized, gamma[j]),
                        beta[j]);
                }
            }
        }

        return output;
    }

    private Tensor<T> ApplyMLP(Tensor<T> input, INumericOperations<T> ops, bool training)
    {
        var shape = input.Shape;
        int batch = shape[0];
        int seqLen = shape[1];

        // Flatten: [batch, seq_len, embed_dim] → [batch * seq_len, embed_dim]
        var flattened = input.Reshape(new[] { batch * seqLen, _embedDim });

        // First linear layer: [batch * seq_len, embed_dim] → [batch * seq_len, mlp_dim]
        var hidden = flattened.MatrixMultiply(_mlp1Weights.Transpose(), ops);
        hidden = AddBias(hidden, _mlp1Bias, ops);

        // GELU activation
        hidden = ApplyGELU(hidden, ops);

        // Dropout
        if (training)
        {
            hidden = ApplyDropout(hidden, _dropoutRate, ops);
        }

        // Second linear layer: [batch * seq_len, mlp_dim] → [batch * seq_len, embed_dim]
        var output = hidden.MatrixMultiply(_mlp2Weights.Transpose(), ops);
        output = AddBias(output, _mlp2Bias, ops);

        // Reshape back: [batch, seq_len, embed_dim]
        return output.Reshape(new[] { batch, seqLen, _embedDim });
    }

    private Tensor<T> ApplyGELU(Tensor<T> input, INumericOperations<T> ops)
    {
        // GELU approximation: 0.5 × x × (1 + tanh(sqrt(2/π) × (x + 0.044715 × x³)))
        var output = new Tensor<T>(input.Shape);
        double sqrt2OverPi = Math.Sqrt(2.0 / Math.PI);
        double coeff = 0.044715;

        for (int i = 0; i < input.Data.Length; i++)
        {
            var x = input.Data[i];
            var x3 = ops.Multiply(x, ops.Square(x));
            var inner = ops.Add(x, ops.Multiply(ops.FromDouble(coeff), x3));
            var scaled = ops.Multiply(ops.FromDouble(sqrt2OverPi), inner);

            // tanh approximation: (e^(2x) - 1) / (e^(2x) + 1)
            var exp2x = ops.Exp(ops.Multiply(ops.FromDouble(2.0), scaled));
            var tanh = ops.Divide(
                ops.Subtract(exp2x, ops.One),
                ops.Add(exp2x, ops.One));

            var factor = ops.Add(ops.One, tanh);
            output.Data[i] = ops.Multiply(
                ops.Multiply(ops.FromDouble(0.5), x),
                factor);
        }

        return output;
    }

    private Tensor<T> ApplyDropout(Tensor<T> input, double rate, INumericOperations<T> ops)
    {
        var output = new Tensor<T>(input.Shape);
        var random = new Random();
        double scale = 1.0 / (1.0 - rate);

        for (int i = 0; i < input.Data.Length; i++)
        {
            if (random.NextDouble() < rate)
            {
                output.Data[i] = ops.Zero;
            }
            else
            {
                output.Data[i] = ops.Multiply(input.Data[i], ops.FromDouble(scale));
            }
        }

        return output;
    }

    private Tensor<T> AddResidual(Tensor<T> input, Tensor<T> residual, INumericOperations<T> ops)
    {
        var output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Data.Length; i++)
        {
            output.Data[i] = ops.Add(input.Data[i], residual.Data[i]);
        }
        return output;
    }

    private Tensor<T> AddBias(Tensor<T> input, Vector<T> bias, INumericOperations<T> ops)
    {
        var shape = input.Shape;
        var output = new Tensor<T>(shape);

        for (int i = 0; i < shape[0]; i++)
        {
            for (int j = 0; j < shape[1]; j++)
            {
                output[i, j] = ops.Add(input[i, j], bias[j]);
            }
        }

        return output;
    }

    private void InitializeParameters(INumericOperations<T> ops)
    {
        var random = new Random(42);

        // Layer norm parameters: gamma = 1, beta = 0
        for (int i = 0; i < _embedDim; i++)
        {
            _norm1Gamma[i] = ops.One;
            _norm1Beta[i] = ops.Zero;
            _norm2Gamma[i] = ops.One;
            _norm2Beta[i] = ops.Zero;
        }

        // MLP weights: Xavier initialization
        double stddev1 = Math.Sqrt(2.0 / (_embedDim + _mlpDim));
        InitializeMatrix(_mlp1Weights, random, stddev1, ops);

        double stddev2 = Math.Sqrt(2.0 / (_mlpDim + _embedDim));
        InitializeMatrix(_mlp2Weights, random, stddev2, ops);

        // MLP biases: zero
        for (int i = 0; i < _mlpDim; i++)
        {
            _mlp1Bias[i] = ops.Zero;
        }

        for (int i = 0; i < _embedDim; i++)
        {
            _mlp2Bias[i] = ops.Zero;
        }
    }

    private void InitializeMatrix(Matrix<T> matrix, Random random, double stddev, INumericOperations<T> ops)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                double u1 = 1.0 - random.NextDouble();
                double u2 = 1.0 - random.NextDouble();
                double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                double value = stddev * z0;
                matrix[i, j] = ops.FromDouble(value);
            }
        }
    }
}
```

### Step 6: Vision Transformer Base Class

#### File: `src/NeuralNetworks/Transformers/Vision/VisionTransformerBase.cs`

```csharp
namespace AiDotNet.NeuralNetworks.Transformers.Vision;

using AiDotNet.Interfaces;
using AiDotNet.Transformers.Embeddings;
using AiDotNet.Transformers.Layers;
using AiDotNet.Mathematics;
using AiDotNet.Validation;

/// <summary>
/// Base class for Vision Transformer models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// This base class provides shared functionality for all Vision Transformer variants:
///
/// 1. **Patch embedding**: Convert image to sequence of patch embeddings
/// 2. **CLS token**: Prepend learnable classification token
/// 3. **Position embeddings**: Add spatial information
/// 4. **Transformer layers**: Stack of encoder blocks
/// 5. **Classification head**: Final MLP for predictions
///
/// Subclasses implement specific variants (ViT, DeiT, Swin) by:
/// - Customizing the initialization
/// - Adding variant-specific components
/// - Overriding certain methods if needed
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public abstract class VisionTransformerBase<T> : IVisionTransformer<T>
{
    protected readonly INumericOperations<T> _ops;
    protected readonly PatchEmbedding<T> _patchEmbedding;
    protected readonly PositionEmbedding<T> _positionEmbedding;
    protected readonly List<TransformerEncoderLayer<T>> _transformerLayers;

    protected readonly Vector<T> _clsToken;
    protected readonly Matrix<T> _classificationHead;
    protected readonly Vector<T> _classificationBias;

    protected readonly int _imageSize;
    protected readonly int _patchSize;
    protected readonly int _embedDim;
    protected readonly int _numLayers;
    protected readonly int _numHeads;
    protected readonly int _mlpDim;
    protected readonly int _numClasses;
    protected readonly double _dropoutRate;

    /// <summary>
    /// Initializes a new instance of the <see cref="VisionTransformerBase{T}"/> class.
    /// </summary>
    protected VisionTransformerBase(
        int imageSize,
        int patchSize,
        int inChannels,
        int numClasses,
        int embedDim,
        int numLayers,
        int numHeads,
        int mlpDim,
        double dropoutRate,
        INumericOperations<T> ops)
    {
        Guard.Positive(imageSize, nameof(imageSize));
        Guard.Positive(patchSize, nameof(patchSize));
        Guard.Positive(inChannels, nameof(inChannels));
        Guard.Positive(numClasses, nameof(numClasses));
        Guard.Positive(embedDim, nameof(embedDim));
        Guard.Positive(numLayers, nameof(numLayers));
        Guard.Positive(numHeads, nameof(numHeads));
        Guard.Positive(mlpDim, nameof(mlpDim));
        Guard.InRange(dropoutRate, 0.0, 1.0, nameof(dropoutRate));
        Guard.NotNull(ops, nameof(ops));

        _imageSize = imageSize;
        _patchSize = patchSize;
        _embedDim = embedDim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _mlpDim = mlpDim;
        _numClasses = numClasses;
        _dropoutRate = dropoutRate;
        _ops = ops;

        // Create patch embedding
        _patchEmbedding = new PatchEmbedding<T>(
            imageSize, patchSize, inChannels, embedDim, ops);

        // Create position embedding (num_patches + 1 for CLS token)
        int numPositions = _patchEmbedding.NumPatches + 1;
        _positionEmbedding = new PositionEmbedding<T>(numPositions, embedDim, ops);

        // Create CLS token
        _clsToken = new Vector<T>(embedDim);
        InitializeClsToken();

        // Create transformer layers
        _transformerLayers = new List<TransformerEncoderLayer<T>>();
        for (int i = 0; i < numLayers; i++)
        {
            _transformerLayers.Add(new TransformerEncoderLayer<T>(
                embedDim, numHeads, mlpDim, dropoutRate, ops));
        }

        // Create classification head
        _classificationHead = new Matrix<T>(numClasses, embedDim);
        _classificationBias = new Vector<T>(numClasses);
        InitializeClassificationHead();
    }

    /// <inheritdoc/>
    public IPatchEmbedding<T> PatchEmbedding => _patchEmbedding;

    /// <inheritdoc/>
    public int NumLayers => _numLayers;

    /// <inheritdoc/>
    public int NumHeads => _numHeads;

    /// <inheritdoc/>
    public int EmbedDim => _embedDim;

    /// <inheritdoc/>
    public int MlpDim => _mlpDim;

    /// <inheritdoc/>
    public virtual Tensor<T> Forward(Tensor<T> input)
    {
        return Forward(input, training: false);
    }

    /// <summary>
    /// Forward pass with training mode option.
    /// </summary>
    public virtual Tensor<T> Forward(Tensor<T> input, bool training)
    {
        Guard.NotNull(input, nameof(input));

        // Extract patches with positions
        var embeddings = ExtractPatches(input);

        // Apply transformer layers
        var transformed = ApplyTransformerLayers(embeddings, training);

        // Take CLS token output (first token)
        var clsOutput = ExtractClsToken(transformed);

        // Classification head
        var logits = ApplyClassificationHead(clsOutput);

        return logits;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> ExtractPatches(Tensor<T> images)
    {
        Guard.NotNull(images, nameof(images));

        // Embed patches: [batch, channels, height, width] → [batch, num_patches, embed_dim]
        var patches = _patchEmbedding.Embed(images, _ops);

        // Prepend CLS token
        var withCls = PrependClsToken(patches);

        // Add position embeddings
        var withPositions = _positionEmbedding.AddPositionEmbeddings(withCls, _ops);

        return withPositions;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> ApplyTransformerLayers(Tensor<T> embeddings)
    {
        return ApplyTransformerLayers(embeddings, training: false);
    }

    /// <summary>
    /// Applies transformer layers with training mode option.
    /// </summary>
    protected virtual Tensor<T> ApplyTransformerLayers(Tensor<T> embeddings, bool training)
    {
        Guard.NotNull(embeddings, nameof(embeddings));

        var output = embeddings;
        foreach (var layer in _transformerLayers)
        {
            output = layer.Forward(output, _ops, training);
        }

        return output;
    }

    private Tensor<T> PrependClsToken(Tensor<T> patches)
    {
        // patches: [batch, num_patches, embed_dim]
        // output: [batch, num_patches + 1, embed_dim]
        var shape = patches.Shape;
        int batch = shape[0];
        int numPatches = shape[1];
        int embedDim = shape[2];

        var output = new Tensor<T>(new[] { batch, numPatches + 1, embedDim });

        for (int b = 0; b < batch; b++)
        {
            // Copy CLS token
            for (int d = 0; d < embedDim; d++)
            {
                output[b, 0, d] = _clsToken[d];
            }

            // Copy patches
            for (int p = 0; p < numPatches; p++)
            {
                for (int d = 0; d < embedDim; d++)
                {
                    output[b, p + 1, d] = patches[b, p, d];
                }
            }
        }

        return output;
    }

    private Tensor<T> ExtractClsToken(Tensor<T> transformed)
    {
        // transformed: [batch, seq_len, embed_dim]
        // output: [batch, embed_dim]
        var shape = transformed.Shape;
        int batch = shape[0];
        int embedDim = shape[2];

        var clsOutput = new Tensor<T>(new[] { batch, embedDim });

        for (int b = 0; b < batch; b++)
        {
            for (int d = 0; d < embedDim; d++)
            {
                clsOutput[b, d] = transformed[b, 0, d]; // First token is CLS
            }
        }

        return clsOutput;
    }

    private Tensor<T> ApplyClassificationHead(Tensor<T> clsOutput)
    {
        // clsOutput: [batch, embed_dim]
        // output: [batch, num_classes]
        var logits = clsOutput.MatrixMultiply(_classificationHead.Transpose(), _ops);

        // Add bias
        var shape = logits.Shape;
        int batch = shape[0];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < _numClasses; c++)
            {
                logits[b, c] = _ops.Add(logits[b, c], _classificationBias[c]);
            }
        }

        return logits;
    }

    private void InitializeClsToken()
    {
        // Initialize with small random values
        var random = new Random(42);
        double stddev = 0.02;

        for (int i = 0; i < _embedDim; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            double value = stddev * z0;
            _clsToken[i] = _ops.FromDouble(value);
        }
    }

    private void InitializeClassificationHead()
    {
        // Xavier initialization
        var random = new Random(42);
        double stddev = Math.Sqrt(2.0 / (_embedDim + _numClasses));

        for (int i = 0; i < _classificationHead.Rows; i++)
        {
            for (int j = 0; j < _classificationHead.Columns; j++)
            {
                double u1 = 1.0 - random.NextDouble();
                double u2 = 1.0 - random.NextDouble();
                double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                double value = stddev * z0;
                _classificationHead[i, j] = _ops.FromDouble(value);
            }
        }

        // Bias initialized to zero
        for (int i = 0; i < _numClasses; i++)
        {
            _classificationBias[i] = _ops.Zero;
        }
    }

    /// <inheritdoc/>
    public abstract void Train(Tensor<T> input, Tensor<T> target);

    /// <inheritdoc/>
    public abstract Tensor<T> Predict(Tensor<T> input);

    /// <inheritdoc/>
    public abstract void Save(string path);

    /// <inheritdoc/>
    public abstract void Load(string path);
}
```

### Step 7: Concrete ViT Implementation

#### File: `src/NeuralNetworks/Transformers/Vision/ViTModel.cs`

```csharp
namespace AiDotNet.NeuralNetworks.Transformers.Vision;

using AiDotNet.Interfaces;
using AiDotNet.Mathematics;
using AiDotNet.Validation;

/// <summary>
/// Implements the original Vision Transformer (ViT) model.
/// </summary>
/// <remarks>
/// <para><b>Paper</b>: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
/// by Dosovitskiy et al. (Google Research, 2020)
///
/// <b>Key Innovation</b>: Apply transformers directly to image patches without convolutions.
///
/// <b>Architecture</b>:
/// - Split image into fixed-size patches (16×16 or 32×32)
/// - Linear embedding of flattened patches
/// - Add learnable position embeddings
/// - Prepend learnable CLS token
/// - Process with standard transformer encoder
/// - Use CLS token output for classification
///
/// <b>Model Variants</b>:
/// - **ViT-Base**: 12 layers, 768 hidden, 12 heads, 86M params
/// - **ViT-Large**: 24 layers, 1024 hidden, 16 heads, 307M params
/// - **ViT-Huge**: 32 layers, 1280 hidden, 16 heads, 632M params
///
/// <b>Training Strategy</b>:
/// 1. Pre-train on large datasets (ImageNet-21K or JFT-300M)
/// 2. Fine-tune on downstream tasks (ImageNet-1K, etc.)
/// 3. Use strong data augmentation and regularization
///
/// <b>For Beginners</b>:
/// ViT showed that the inductive biases of CNNs (convolutions, pooling) are not
/// necessary for excellent image recognition when you have enough data and compute.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ViTModel<T> : VisionTransformerBase<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ViTModel{T}"/> class.
    /// </summary>
    /// <param name="imageSize">Size of input images (assumed square, e.g., 224).</param>
    /// <param name="patchSize">Size of patches (e.g., 16 for 16×16 patches).</param>
    /// <param name="inChannels">Number of input channels (e.g., 3 for RGB).</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="embedDim">Embedding dimension (e.g., 768 for ViT-Base).</param>
    /// <param name="numLayers">Number of transformer layers (e.g., 12 for ViT-Base).</param>
    /// <param name="numHeads">Number of attention heads (e.g., 12 for ViT-Base).</param>
    /// <param name="mlpDim">MLP hidden dimension (typically 4 × embed_dim).</param>
    /// <param name="dropoutRate">Dropout rate for regularization (e.g., 0.1).</param>
    /// <param name="ops">Numeric operations provider.</param>
    public ViTModel(
        int imageSize,
        int patchSize,
        int inChannels,
        int numClasses,
        int embedDim,
        int numLayers,
        int numHeads,
        int mlpDim,
        double dropoutRate,
        INumericOperations<T> ops)
        : base(
            imageSize,
            patchSize,
            inChannels,
            numClasses,
            embedDim,
            numLayers,
            numHeads,
            mlpDim,
            dropoutRate,
            ops)
    {
    }

    /// <summary>
    /// Creates a ViT-Base model configuration.
    /// </summary>
    /// <param name="imageSize">Image size (default: 224).</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="ops">Numeric operations provider.</param>
    /// <returns>A ViT-Base model instance.</returns>
    public static ViTModel<T> CreateBase(
        int imageSize,
        int numClasses,
        INumericOperations<T> ops)
    {
        return new ViTModel<T>(
            imageSize: imageSize,
            patchSize: 16,
            inChannels: 3,
            numClasses: numClasses,
            embedDim: 768,
            numLayers: 12,
            numHeads: 12,
            mlpDim: 3072, // 4 × 768
            dropoutRate: 0.1,
            ops: ops);
    }

    /// <summary>
    /// Creates a ViT-Large model configuration.
    /// </summary>
    public static ViTModel<T> CreateLarge(
        int imageSize,
        int numClasses,
        INumericOperations<T> ops)
    {
        return new ViTModel<T>(
            imageSize: imageSize,
            patchSize: 16,
            inChannels: 3,
            numClasses: numClasses,
            embedDim: 1024,
            numLayers: 24,
            numHeads: 16,
            mlpDim: 4096, // 4 × 1024
            dropoutRate: 0.1,
            ops: ops);
    }

    /// <summary>
    /// Creates a ViT-Huge model configuration.
    /// </summary>
    public static ViTModel<T> CreateHuge(
        int imageSize,
        int numClasses,
        INumericOperations<T> ops)
    {
        return new ViTModel<T>(
            imageSize: imageSize,
            patchSize: 14,
            inChannels: 3,
            numClasses: numClasses,
            embedDim: 1280,
            numLayers: 32,
            numHeads: 16,
            mlpDim: 5120, // 4 × 1280
            dropoutRate: 0.1,
            ops: ops);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        Guard.NotNull(input, nameof(input));
        Guard.NotNull(target, nameof(target));

        // Forward pass in training mode
        var predictions = Forward(input, training: true);

        // Compute loss (cross-entropy)
        var loss = ComputeCrossEntropyLoss(predictions, target);

        // Backward pass (gradient computation)
        // TODO: Implement backpropagation through transformer layers
        // This requires tracking intermediate activations and implementing
        // gradient computation for each layer

        // Update parameters
        // TODO: Implement optimizer (Adam, AdamW, etc.)
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        Guard.NotNull(input, nameof(input));

        // Forward pass in inference mode
        var logits = Forward(input, training: false);

        // Apply softmax to get probabilities
        return ApplySoftmax(logits);
    }

    /// <inheritdoc/>
    public override void Save(string path)
    {
        Guard.NotNullOrEmpty(path, nameof(path));

        // TODO: Implement model serialization
        // Save:
        // - Model configuration (image_size, patch_size, embed_dim, etc.)
        // - All learned parameters (patch embedding, position embeddings, transformer layers, classification head)
        // - Optimizer state (if needed for resuming training)
    }

    /// <inheritdoc/>
    public override void Load(string path)
    {
        Guard.NotNullOrEmpty(path, nameof(path));

        // TODO: Implement model deserialization
        // Load all saved parameters and restore model state
    }

    private T ComputeCrossEntropyLoss(Tensor<T> predictions, Tensor<T> targets)
    {
        // Cross-entropy loss: -sum(target * log(softmax(prediction)))
        var softmax = ApplySoftmax(predictions);
        T loss = _ops.Zero;

        var predShape = predictions.Shape;
        int batch = predShape[0];
        int numClasses = predShape[1];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < numClasses; c++)
            {
                var target = targets[b, c];
                var pred = softmax[b, c];

                // Avoid log(0)
                if (_ops.GreaterThan(pred, _ops.Zero))
                {
                    var logPred = _ops.Log(pred);
                    var term = _ops.Multiply(target, logPred);
                    loss = _ops.Add(loss, term);
                }
            }
        }

        // Negate and normalize by batch size
        loss = _ops.Negate(loss);
        loss = _ops.Divide(loss, _ops.FromDouble(batch));

        return loss;
    }

    private Tensor<T> ApplySoftmax(Tensor<T> logits)
    {
        var shape = logits.Shape;
        int batch = shape[0];
        int numClasses = shape[1];
        var softmax = new Tensor<T>(shape);

        for (int b = 0; b < batch; b++)
        {
            // Find max for numerical stability
            T maxLogit = logits[b, 0];
            for (int c = 1; c < numClasses; c++)
            {
                if (_ops.GreaterThan(logits[b, c], maxLogit))
                {
                    maxLogit = logits[b, c];
                }
            }

            // Compute exp(x - max) and sum
            T sum = _ops.Zero;
            var exps = new T[numClasses];
            for (int c = 0; c < numClasses; c++)
            {
                var shifted = _ops.Subtract(logits[b, c], maxLogit);
                exps[c] = _ops.Exp(shifted);
                sum = _ops.Add(sum, exps[c]);
            }

            // Normalize
            for (int c = 0; c < numClasses; c++)
            {
                softmax[b, c] = _ops.Divide(exps[c], sum);
            }
        }

        return softmax;
    }
}
```

---

## Testing Strategy

### Unit Tests

#### Test 1: Patch Embedding

```csharp
namespace AiDotNetTests.UnitTests.Transformers.Embeddings;

using AiDotNet.Transformers.Embeddings;
using AiDotNet.Mathematics;
using Xunit;

public class PatchEmbeddingTests
{
    [Fact]
    public void Constructor_ValidParameters_InitializesCorrectly()
    {
        // Arrange
        var ops = new DoubleNumericOperations();

        // Act
        var patchEmbed = new PatchEmbedding<double>(
            imageSize: 224,
            patchSize: 16,
            inChannels: 3,
            embedDim: 768,
            ops: ops);

        // Assert
        Assert.Equal(196, patchEmbed.NumPatches); // (224/16)²
        Assert.Equal(768, patchEmbed.EmbedDim);
        Assert.Equal(16, patchEmbed.PatchSize);
    }

    [Fact]
    public void Embed_ValidInput_ReturnsCorrectShape()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var patchEmbed = new PatchEmbedding<double>(224, 16, 3, 768, ops);
        var input = new Tensor<double>(new[] { 2, 3, 224, 224 }); // Batch of 2

        // Act
        var output = patchEmbed.Embed(input, ops);

        // Assert
        Assert.Equal(new[] { 2, 196, 768 }, output.Shape);
    }

    [Theory]
    [InlineData(200)] // Not divisible by 16
    [InlineData(100)]
    public void Constructor_ImageSizeNotDivisibleByPatchSize_ThrowsException(int imageSize)
    {
        // Arrange
        var ops = new DoubleNumericOperations();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new PatchEmbedding<double>(imageSize, 16, 3, 768, ops));
    }
}
```

#### Test 2: Position Embeddings

```csharp
namespace AiDotNetTests.UnitTests.Transformers.Embeddings;

using AiDotNet.Transformers.Embeddings;
using AiDotNet.Mathematics;
using Xunit;

public class PositionEmbeddingTests
{
    [Fact]
    public void AddPositionEmbeddings_ValidInput_ReturnsCorrectShape()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var posEmbed = new PositionEmbedding<double>(197, 768, ops); // 196 patches + 1 CLS
        var input = new Tensor<double>(new[] { 2, 197, 768 });

        // Act
        var output = posEmbed.AddPositionEmbeddings(input, ops);

        // Assert
        Assert.Equal(new[] { 2, 197, 768 }, output.Shape);
    }

    [Fact]
    public void AddPositionEmbeddings_InputSequenceTooLong_ThrowsException()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var posEmbed = new PositionEmbedding<double>(100, 768, ops);
        var input = new Tensor<double>(new[] { 2, 200, 768 }); // Longer than 100

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            posEmbed.AddPositionEmbeddings(input, ops));
    }

    [Fact]
    public void GetPositionEmbedding_ValidPosition_ReturnsCorrectDimension()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var posEmbed = new PositionEmbedding<double>(197, 768, ops);

        // Act
        var embedding = posEmbed.GetPositionEmbedding(0);

        // Assert
        Assert.Equal(768, embedding.Length);
    }
}
```

#### Test 3: Multi-Head Self-Attention

```csharp
namespace AiDotNetTests.UnitTests.Transformers.Attention;

using AiDotNet.Transformers.Attention;
using AiDotNet.Mathematics;
using Xunit;

public class MultiHeadSelfAttentionTests
{
    [Fact]
    public void Forward_ValidInput_ReturnsCorrectShape()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var attention = new MultiHeadSelfAttention<double>(768, 12, ops);
        var input = new Tensor<double>(new[] { 2, 197, 768 });

        // Act
        var output = attention.Forward(input, ops);

        // Assert
        Assert.Equal(new[] { 2, 197, 768 }, output.Shape);
    }

    [Theory]
    [InlineData(100, 3)] // 100 not divisible by 3
    [InlineData(768, 5)] // 768 not divisible by 5
    public void Constructor_EmbedDimNotDivisibleByHeads_ThrowsException(int embedDim, int numHeads)
    {
        // Arrange
        var ops = new DoubleNumericOperations();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new MultiHeadSelfAttention<double>(embedDim, numHeads, ops));
    }

    [Fact]
    public void Forward_AttentionPreservesEmbeddingDimension()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var attention = new MultiHeadSelfAttention<double>(768, 12, ops);
        var input = new Tensor<double>(new[] { 1, 10, 768 });

        // Fill with known values
        for (int i = 0; i < input.Data.Length; i++)
        {
            input.Data[i] = 1.0;
        }

        // Act
        var output = attention.Forward(input, ops);

        // Assert
        Assert.Equal(768, output.Shape[2]);
        // Output should not be all zeros (attention did something)
        Assert.True(output.Data.Any(x => Math.Abs(x) > 0.001));
    }
}
```

#### Test 4: ViT Model

```csharp
namespace AiDotNetTests.UnitTests.NeuralNetworks.Transformers.Vision;

using AiDotNet.NeuralNetworks.Transformers.Vision;
using AiDotNet.Mathematics;
using Xunit;

public class ViTModelTests
{
    [Fact]
    public void CreateBase_ReturnsValidModel()
    {
        // Arrange
        var ops = new DoubleNumericOperations();

        // Act
        var model = ViTModel<double>.CreateBase(224, 1000, ops);

        // Assert
        Assert.NotNull(model);
        Assert.Equal(12, model.NumLayers);
        Assert.Equal(12, model.NumHeads);
        Assert.Equal(768, model.EmbedDim);
        Assert.Equal(3072, model.MlpDim);
    }

    [Fact]
    public void Forward_ValidInput_ReturnsCorrectShape()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var model = ViTModel<double>.CreateBase(224, 1000, ops);
        var input = new Tensor<double>(new[] { 2, 3, 224, 224 });

        // Act
        var output = model.Forward(input);

        // Assert
        Assert.Equal(new[] { 2, 1000 }, output.Shape); // [batch, num_classes]
    }

    [Fact]
    public void Predict_ValidInput_ReturnsProbabilities()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var model = ViTModel<double>.CreateBase(224, 1000, ops);
        var input = new Tensor<double>(new[] { 1, 3, 224, 224 });

        // Act
        var probabilities = model.Predict(input);

        // Assert
        Assert.Equal(new[] { 1, 1000 }, probabilities.Shape);

        // Probabilities should sum to ~1.0
        double sum = 0;
        for (int i = 0; i < 1000; i++)
        {
            sum += probabilities[0, i];
        }
        Assert.True(Math.Abs(sum - 1.0) < 0.01);
    }

    [Fact]
    public void ExtractPatches_ValidInput_ReturnsCorrectNumberOfPatches()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var model = ViTModel<double>.CreateBase(224, 1000, ops);
        var input = new Tensor<double>(new[] { 1, 3, 224, 224 });

        // Act
        var patches = model.ExtractPatches(input);

        // Assert
        Assert.Equal(new[] { 1, 197, 768 }, patches.Shape); // 196 patches + 1 CLS token
    }
}
```

### Integration Tests

```csharp
namespace AiDotNetTests.IntegrationTests.Transformers.Vision;

using AiDotNet.NeuralNetworks.Transformers.Vision;
using AiDotNet.Mathematics;
using Xunit;

public class ViTIntegrationTests
{
    [Fact]
    public void EndToEnd_Training_Works()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var model = ViTModel<double>.CreateBase(224, 10, ops);

        var images = new Tensor<double>(new[] { 4, 3, 224, 224 });
        var labels = new Tensor<double>(new[] { 4, 10 });

        // Initialize with random data
        var random = new Random(42);
        for (int i = 0; i < images.Data.Length; i++)
        {
            images.Data[i] = random.NextDouble();
        }

        // One-hot encode labels
        for (int i = 0; i < 4; i++)
        {
            int labelIdx = random.Next(10);
            labels[i, labelIdx] = 1.0;
        }

        // Act - Single training step
        model.Train(images, labels);

        // Assert - Model should still work after training step
        var output = model.Forward(images);
        Assert.Equal(new[] { 4, 10 }, output.Shape);
    }

    [Fact]
    public void EndToEnd_SaveLoad_PreservesModel()
    {
        // Arrange
        var ops = new DoubleNumericOperations();
        var model1 = ViTModel<double>.CreateBase(224, 10, ops);
        var input = new Tensor<double>(new[] { 1, 3, 224, 224 });

        // Get prediction before saving
        var predictionBefore = model1.Predict(input);

        // Act - Save and load
        string tempPath = Path.Combine(Path.GetTempPath(), "vit_model_test.bin");
        model1.Save(tempPath);

        var model2 = ViTModel<double>.CreateBase(224, 10, ops);
        model2.Load(tempPath);

        // Get prediction after loading
        var predictionAfter = model2.Predict(input);

        // Assert - Predictions should match
        Assert.Equal(predictionBefore.Shape, predictionAfter.Shape);
        for (int i = 0; i < predictionBefore.Data.Length; i++)
        {
            Assert.True(Math.Abs(predictionBefore.Data[i] - predictionAfter.Data[i]) < 0.001);
        }

        // Cleanup
        File.Delete(tempPath);
    }
}
```

---

## Training Strategy

### Pre-training

```csharp
/// <summary>
/// Pre-trains a Vision Transformer on a large dataset.
/// </summary>
/// <remarks>
/// <b>Pre-training Strategy</b>:
///
/// 1. **Dataset**: ImageNet-21K (14M images, 21K classes) or larger
/// 2. **Batch Size**: Large (e.g., 4096) using gradient accumulation if needed
/// 3. **Optimizer**: AdamW with weight decay
/// 4. **Learning Rate**: Warmup then cosine decay
///    - Warmup: 10K steps, linear increase to peak LR
///    - Peak LR: 0.001 for ViT-Base
///    - Cosine decay to 0 over remaining steps
/// 5. **Data Augmentation**:
///    - RandAugment
///    - Random cropping and resizing
///    - Horizontal flipping
///    - Color jitter
/// 6. **Regularization**:
///    - Dropout: 0.1
///    - Stochastic depth (drop path): 0.1
///    - Label smoothing: 0.1
/// 7. **Duration**: 300 epochs on ImageNet-21K
///
/// <b>Computational Requirements</b>:
/// - ViT-Base: ~8 TPUv3 cores, ~7 days
/// - ViT-Large: ~32 TPUv3 cores, ~14 days
/// </remarks>
public class ViTPreTrainer<T>
{
    // TODO: Implement pre-training loop with:
    // - Data loading and augmentation
    // - Optimizer (AdamW)
    // - Learning rate scheduling
    // - Mixed precision training
    // - Gradient clipping
    // - Checkpointing
    // - Logging and monitoring
}
```

### Fine-tuning

```csharp
/// <summary>
/// Fine-tunes a pre-trained Vision Transformer on a downstream task.
/// </summary>
/// <remarks>
/// <b>Fine-tuning Strategy</b>:
///
/// 1. **Initialization**: Load pre-trained weights
/// 2. **Resolution**: Can use higher resolution than pre-training (e.g., 384 instead of 224)
///    - Interpolate position embeddings to new resolution
/// 3. **Learning Rate**: Lower than pre-training (e.g., 0.0001)
/// 4. **Batch Size**: Smaller (e.g., 512)
/// 5. **Duration**: Shorter (e.g., 20-100 epochs)
/// 6. **Regularization**: Less aggressive than pre-training
///
/// <b>Example Fine-tuning</b>:
/// ```
/// var pretrainedModel = ViTModel<double>.CreateBase(224, 21000, ops);
/// pretrainedModel.Load("vit_base_imagenet21k.bin");
///
/// var finetuneModel = new ViTModel<double>(
///     imageSize: 384,  // Higher resolution
///     patchSize: 16,
///     inChannels: 3,
///     numClasses: 1000,  // ImageNet-1K
///     embedDim: 768,
///     numLayers: 12,
///     numHeads: 12,
///     mlpDim: 3072,
///     dropoutRate: 0.0,  // No dropout during fine-tuning
///     ops: ops);
///
/// // Transfer weights (except classification head)
/// finetuneModel.TransferWeights(pretrainedModel);
///
/// // Fine-tune on ImageNet-1K
/// finetuneModel.FineTune(imagenet1kDataset, epochs: 20, learningRate: 0.0001);
/// ```
/// </remarks>
public class ViTFineTuner<T>
{
    // TODO: Implement fine-tuning with:
    // - Weight transfer from pre-trained model
    // - Position embedding interpolation for different resolutions
    // - Lower learning rates
    // - Optional layer-wise learning rate decay
}
```

---

## Common Pitfalls and Debugging

### Pitfall 1: Incorrect Patch Extraction
**Problem**: Patches extracted in wrong order or with wrong dimensions.
**Solution**: Verify patch extraction with visualization. Ensure row-major order.

### Pitfall 2: Position Embedding Mismatch
**Problem**: Position embeddings don't match sequence length.
**Solution**: Always use `num_patches + 1` for position embeddings (include CLS token).

### Pitfall 3: Attention Dimension Errors
**Problem**: Embedding dimension not divisible by number of heads.
**Solution**: Ensure `embed_dim % num_heads == 0`.

### Pitfall 4: Numerical Instability in Softmax
**Problem**: Overflow/underflow in attention computation.
**Solution**: Always subtract max before computing exp in softmax.

### Pitfall 5: Forgetting CLS Token
**Problem**: Using average of all tokens instead of CLS token for classification.
**Solution**: Always extract first token (index 0) after transformer layers.

---

## Performance Benchmarks

### Expected Accuracy (with proper training)

| Model | Dataset | Top-1 Accuracy | Top-5 Accuracy |
|-------|---------|----------------|----------------|
| ViT-Base/16 (pre-trained on ImageNet-21K) | ImageNet-1K | 84.5% | 97.1% |
| ViT-Large/16 (pre-trained on ImageNet-21K) | ImageNet-1K | 86.5% | 98.0% |
| ViT-Huge/14 (pre-trained on JFT-300M) | ImageNet-1K | 88.5% | 98.5% |

### Computational Requirements

| Model | Parameters | FLOPs (per image) | Memory |
|-------|-----------|-------------------|---------|
| ViT-Base/16 | 86M | 17.6 GFLOPs | ~3.3 GB |
| ViT-Large/16 | 307M | 61.6 GFLOPs | ~12 GB |
| ViT-Huge/14 | 632M | 167.4 GFLOPs | ~25 GB |

---

## Next Steps

After implementing the basic ViT, consider:

1. **DeiT Implementation**: Add distillation token and training strategy
2. **Swin Transformer**: Implement hierarchical architecture with shifted windows
3. **Advanced Features**:
   - Mixed precision training (FP16/BF16)
   - Gradient checkpointing for memory efficiency
   - Multi-scale training
   - Test-time augmentation
4. **Optimizations**:
   - Flash Attention for faster attention computation
   - Fused kernels for LayerNorm and GELU
   - Model quantization for deployment

---

## Resources

- [Original ViT Paper](https://arxiv.org/abs/2010.11929)
- [DeiT Paper](https://arxiv.org/abs/2012.12877)
- [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
- [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
