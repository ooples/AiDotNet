# Junior Developer Implementation Guide: Issue #262
## Latent Diffusion with Text Conditioning (CLIP/Transformer)

### Overview
This guide builds on Issue #261 (core diffusion) to add text-to-image generation capabilities. You'll implement latent diffusion (working in compressed space) with text conditioning using CLIP-style encoders.

---

## Understanding Latent Diffusion with Text Conditioning

### What is Latent Diffusion?

**Problem with Pixel-Space Diffusion:**
- Generating a 512x512 RGB image = 786,432 values to denoise
- Very slow and memory-intensive
- Most computation wasted on imperceptible details

**Solution: Latent Diffusion**
- Compress image to smaller "latent" representation using VAE
- Run diffusion in this compressed space (e.g., 64x64x4 = 16,384 values)
- Decompress back to pixel space
- **48x faster with similar quality!**

**Real-World Analogy:**
Think of it like architectural blueprints:
- Pixel space = Building the actual house (slow, expensive)
- Latent space = Drawing blueprints (fast, cheap)
- VAE encoder = Creating blueprints from house
- VAE decoder = Building house from blueprints
- Diffusion = Editing/creating new blueprints

### What is Text Conditioning?

**Goal:** Generate images that match text descriptions

**Process:**
1. **Text Encoding**: Convert "a cat wearing sunglasses" → vector of numbers
2. **Cross-Attention**: Let the diffusion model "look at" the text while generating
3. **Classifier-Free Guidance**: Make the model follow text more strongly

**Key Components:**

#### 1. Text Encoder (CLIP-style)
- Tokenizer: Splits text into words/subwords
- Transformer: Processes tokens → embeddings
- Output: Text representation the model understands

#### 2. Cross-Attention
- **Self-attention**: Image features attend to other image features
- **Cross-attention**: Image features attend to text features
- Allows model to say "I'm generating fur texture, let me check the text mentions 'cat'"

#### 3. Classifier-Free Guidance (CFG)
- Generates with AND without text
- Emphasizes the difference
- guidance_scale controls strength (7.5 is typical)

**Formula:**
```
output = unconditional_output + scale * (conditional_output - unconditional_output)
```

---

## Architecture Overview

### File Structure
```
src/
├── Interfaces/
│   ├── ITextEncoder.cs          # Text encoding interface
│   ├── ITokenizer.cs             # Tokenization interface
│   ├── ILatentDiffusionModel.cs  # Latent diffusion interface
│   └── IVAE.cs                   # VAE interface
├── Models/
│   └── Generative/
│       └── Diffusion/
│           ├── LatentDiffusionModel.cs     # Main implementation
│           └── TextConditionedDiffusion.cs # Text-conditioned variant
├── Encoders/
│   ├── CLIPTextEncoder.cs        # CLIP-style text encoder
│   └── SimpleTokenizer.cs        # Basic tokenizer
├── NeuralNetworks/
│   ├── Architectures/
│   │   ├── VAE/
│   │   │   ├── VAEEncoder.cs    # Image → latent
│   │   │   ├── VAEDecoder.cs    # Latent → image
│   │   │   └── VAE.cs            # Combined VAE
│   │   └── UNet/
│   │       ├── UNetModel.cs             # U-Net architecture
│   │       ├── ResNetBlock.cs           # Residual blocks
│   │       ├── AttentionBlock.cs        # Self-attention
│   │       └── CrossAttentionBlock.cs   # NEW: Text cross-attention
│   └── Layers/
│       └── CrossAttentionLayer.cs # NEW: Cross-attention implementation
```

### Component Relationships
```
User Input: "A cat wearing sunglasses"
    ↓
[Tokenizer] → [CLIP Encoder] → Text Embeddings
                                      ↓
Random Noise → [U-Net with Cross-Attention] → Denoised Latents
                      ↑ (attends to text)
                      Scheduler controls timesteps
    ↓
[VAE Decoder] → Final Image
```

---

## Step-by-Step Implementation

### Step 1: Create Tokenizer Interface and Implementation

#### File: `src/Interfaces/ITokenizer.cs`
```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for text tokenization.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> A tokenizer splits text into pieces the model can understand.
///
/// Think of it like translating a sentence:
/// - Input: "Hello world"
/// - Tokenizer breaks it into: ["Hello", "world"]
/// - Converts to numbers: [2534, 1847]
/// - Model processes numbers, not text
///
/// Why tokenization?
/// - Neural networks work with numbers, not text
/// - Tokenization creates a fixed vocabulary (e.g., 50,000 words)
/// - Handles unknown words by breaking into subwords
///
/// Example:
/// - "cat" → token ID 1234
/// - "cats" → token ID 1235
/// - "unbelievable" → ["un", "believable"] → [567, 890]
/// </remarks>
public interface ITokenizer
{
    /// <summary>
    /// Tokenizes text into a sequence of token IDs.
    /// </summary>
    /// <param name="text">The input text.</param>
    /// <returns>Array of token IDs.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Converts text to numbers.
    ///
    /// Steps:
    /// 1. Lowercase and clean text (optional)
    /// 2. Split into words or subwords
    /// 3. Look up each word in vocabulary
    /// 4. Return array of IDs
    ///
    /// Example:
    /// Tokenize("A cute cat") → [49406, 7846, 2368, 49407]
    /// [49406 = START, 7846 = cute, 2368 = cat, 49407 = END]
    /// </remarks>
    int[] Tokenize(string text);

    /// <summary>
    /// Converts token IDs back to text.
    /// </summary>
    /// <param name="tokenIds">Array of token IDs.</param>
    /// <returns>Decoded text.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Reverse of tokenization.
    ///
    /// Useful for debugging and verification.
    /// </remarks>
    string Detokenize(int[] tokenIds);

    /// <summary>
    /// Gets the vocabulary size.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Total number of unique tokens.
    ///
    /// Typical values:
    /// - Small: 10,000 tokens
    /// - Medium: 50,000 tokens (common)
    /// - Large: 100,000+ tokens
    /// </remarks>
    int VocabularySize { get; }

    /// <summary>
    /// Gets the maximum sequence length.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Maximum number of tokens in a prompt.
    ///
    /// CLIP uses 77 tokens. If text is longer, it's truncated.
    /// </remarks>
    int MaxLength { get; }
}
```

#### File: `src/Encoders/SimpleTokenizer.cs`
```csharp
using AiDotNet.Interfaces;

namespace AiDotNet.Encoders;

/// <summary>
/// Simple tokenizer for text encoding.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Basic word-level tokenizer.
///
/// This is a simplified implementation for demonstration.
/// Production systems use BPE (Byte-Pair Encoding) like CLIP.
///
/// Limitations:
/// - Fixed vocabulary
/// - No subword handling
/// - English only
///
/// For production, consider using:
/// - HuggingFace tokenizers
/// - SentencePiece
/// - CLIP's BPE tokenizer
/// </remarks>
public class SimpleTokenizer : ITokenizer
{
    private readonly Dictionary<string, int> _vocabulary;
    private readonly Dictionary<int, string> _reverseVocabulary;
    private readonly int _maxLength;

    // Special tokens
    private const int START_TOKEN = 49406;
    private const int END_TOKEN = 49407;
    private const int PAD_TOKEN = 0;

    public int VocabularySize => _vocabulary.Count;
    public int MaxLength => _maxLength;

    /// <summary>
    /// Initializes the tokenizer with a vocabulary.
    /// </summary>
    /// <param name="maxLength">Maximum sequence length. Default: 77 (CLIP standard)</param>
    /// <remarks>
    /// <b>For Beginners:</b> Sets up the tokenizer.
    ///
    /// In production, vocabulary would be loaded from a file.
    /// Here we create a minimal vocabulary for demonstration.
    /// </remarks>
    public SimpleTokenizer(int maxLength = 77)
    {
        _maxLength = maxLength;
        _vocabulary = new Dictionary<string, int>();
        _reverseVocabulary = new Dictionary<int, string>();

        // Build a minimal vocabulary (in production, load from file)
        BuildVocabulary();
    }

    private void BuildVocabulary()
    {
        // Special tokens
        AddToken("<|startoftext|>", START_TOKEN);
        AddToken("<|endoftext|>", END_TOKEN);
        AddToken("<|pad|>", PAD_TOKEN);

        // Common words (minimal set for demonstration)
        string[] commonWords = new[]
        {
            "a", "an", "the", "of", "in", "on", "at", "to", "for", "with",
            "cat", "dog", "bird", "fish", "animal", "pet",
            "red", "blue", "green", "yellow", "orange", "purple",
            "big", "small", "cute", "beautiful", "happy", "sad",
            "wearing", "holding", "sitting", "standing", "running",
            "hat", "sunglasses", "shirt", "shoes",
            "photo", "image", "picture", "painting", "drawing"
        };

        int tokenId = 1000; // Start regular tokens at 1000
        foreach (var word in commonWords)
        {
            AddToken(word, tokenId++);
        }
    }

    private void AddToken(string word, int tokenId)
    {
        _vocabulary[word] = tokenId;
        _reverseVocabulary[tokenId] = word;
    }

    public int[] Tokenize(string text)
    {
        if (string.IsNullOrEmpty(text))
            return new[] { START_TOKEN, END_TOKEN };

        // Lowercase and split
        var words = text.ToLower().Split(new[] { ' ', '\t', '\n' }, StringSplitOptions.RemoveEmptyEntries);

        // Convert to tokens
        var tokens = new List<int> { START_TOKEN };

        foreach (var word in words)
        {
            if (tokens.Count >= _maxLength - 1) // Leave room for END_TOKEN
                break;

            // Clean word (remove punctuation)
            string cleanWord = new string(word.Where(char.IsLetterOrDigit).ToArray());

            if (_vocabulary.TryGetValue(cleanWord, out int tokenId))
            {
                tokens.Add(tokenId);
            }
            else
            {
                // Unknown word - in production, use BPE to split into subwords
                // For now, use a special UNK token or skip
                // tokens.Add(UNK_TOKEN);
            }
        }

        tokens.Add(END_TOKEN);

        // Pad to max length
        while (tokens.Count < _maxLength)
        {
            tokens.Add(PAD_TOKEN);
        }

        return tokens.ToArray();
    }

    public string Detokenize(int[] tokenIds)
    {
        var words = new List<string>();

        foreach (var tokenId in tokenIds)
        {
            if (tokenId == START_TOKEN || tokenId == END_TOKEN || tokenId == PAD_TOKEN)
                continue;

            if (_reverseVocabulary.TryGetValue(tokenId, out string? word))
            {
                words.Add(word);
            }
        }

        return string.Join(" ", words);
    }
}
```

---

### Step 2: Create Text Encoder Interface and Implementation

#### File: `src/Interfaces/ITextEncoder.cs`
```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for encoding text into embeddings.
/// </summary>
/// <typeparam name="T">Numeric type for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Converts text into vectors that diffusion models can use.
///
/// Think of it like a translator:
/// - Input: "a cat wearing sunglasses" (human language)
/// - Output: Matrix of numbers (machine language)
/// - The numbers capture the meaning of the text
///
/// Process:
/// 1. Tokenize text → [token IDs]
/// 2. Embed tokens → [embedding vectors]
/// 3. Transformer processes embeddings → [contextualized embeddings]
/// 4. Return final representation
///
/// CLIP vs Other Encoders:
/// - CLIP: Trained on image-text pairs, understands visual concepts
/// - BERT: General language understanding
/// - T5: Sequence-to-sequence tasks
///
/// For text-to-image, CLIP is preferred because it was trained to align
/// images and text in the same embedding space.
/// </remarks>
public interface ITextEncoder<T>
{
    /// <summary>
    /// Encodes text into embeddings for conditioning.
    /// </summary>
    /// <param name="text">The input text prompt.</param>
    /// <returns>Text embeddings tensor [1, seq_len, hidden_dim].</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Main method to convert text to embeddings.
    ///
    /// Output shape:
    /// - Batch dimension: 1 (single prompt)
    /// - Sequence length: 77 tokens (CLIP standard)
    /// - Hidden dimension: 768 or 1024 (embedding size)
    ///
    /// Example:
    /// Input: "a beautiful cat"
    /// Output: Tensor<double> with shape [1, 77, 768]
    /// Each of 77 positions has a 768-dimensional vector
    /// </remarks>
    Tensor<T> Encode(string text);

    /// <summary>
    /// Encodes batch of texts.
    /// </summary>
    /// <param name="texts">Array of text prompts.</param>
    /// <returns>Text embeddings tensor [batch_size, seq_len, hidden_dim].</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Encode multiple prompts at once for efficiency.
    ///
    /// Example:
    /// Input: ["a cat", "a dog", "a bird"]
    /// Output: Tensor<double> with shape [3, 77, 768]
    /// </remarks>
    Tensor<T> EncodeBatch(string[] texts);

    /// <summary>
    /// Gets the tokenizer used by this encoder.
    /// </summary>
    ITokenizer Tokenizer { get; }

    /// <summary>
    /// Gets the hidden dimension size.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Size of embedding vectors.
    ///
    /// Common sizes:
    /// - CLIP ViT-B: 512
    /// - CLIP ViT-L: 768
    /// - T5-Large: 1024
    /// </remarks>
    int HiddenDim { get; }
}
```

#### File: `src/Encoders/CLIPTextEncoder.cs`
```csharp
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Mathematics;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Encoders;

/// <summary>
/// CLIP-style text encoder for diffusion conditioning.
/// </summary>
/// <typeparam name="T">Numeric type for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Converts text to embeddings using CLIP architecture.
///
/// CLIP (Contrastive Language-Image Pre-training) from OpenAI:
/// - Trained on 400M image-text pairs
/// - Learns to align images and text in same embedding space
/// - Perfect for text-to-image because it understands visual concepts
///
/// Architecture:
/// 1. Token Embedding: Convert token IDs to vectors
/// 2. Positional Encoding: Add position information
/// 3. Transformer Layers: Process with self-attention (12 layers)
/// 4. Output: Contextualized embeddings
///
/// Simplified from: "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)
/// </remarks>
public class CLIPTextEncoder<T> : ITextEncoder<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly ITokenizer _tokenizer;
    private readonly int _hiddenDim;
    private readonly int _numLayers;
    private readonly int _numHeads;

    // Embeddings
    private Tensor<T> _tokenEmbedding;      // [vocab_size, hidden_dim]
    private Tensor<T> _positionalEmbedding; // [max_length, hidden_dim]

    // Transformer layers (simplified - in production, use TransformerEncoderLayer)
    private readonly List<ILayer<T>> _transformerLayers;

    public ITokenizer Tokenizer => _tokenizer;
    public int HiddenDim => _hiddenDim;

    /// <summary>
    /// Initializes the CLIP text encoder.
    /// </summary>
    /// <param name="tokenizer">Tokenizer for text processing.</param>
    /// <param name="hiddenDim">
    /// Embedding dimension. Default: 768 (CLIP ViT-L).
    /// CLIP ViT-B uses 512, CLIP ViT-L uses 768.
    /// </param>
    /// <param name="numLayers">
    /// Number of transformer layers. Default: 12 (CLIP standard).
    /// More layers = more capacity but slower.
    /// </param>
    /// <param name="numHeads">
    /// Number of attention heads. Default: 12.
    /// Must divide hiddenDim evenly (768 / 12 = 64 per head).
    /// </param>
    /// <remarks>
    /// <b>For Beginners:</b> Sets up the text encoder.
    ///
    /// Defaults match CLIP ViT-L configuration.
    /// Adjust for speed vs quality tradeoff.
    /// </remarks>
    public CLIPTextEncoder(
        ITokenizer? tokenizer = null,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12)
    {
        _tokenizer = tokenizer ?? new SimpleTokenizer();
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _numHeads = numHeads;

        if (_hiddenDim % _numHeads != 0)
            throw new ArgumentException($"hiddenDim ({hiddenDim}) must be divisible by numHeads ({numHeads})");

        // Initialize embeddings
        InitializeEmbeddings();

        // Initialize transformer layers (placeholder)
        _transformerLayers = new List<ILayer<T>>();
        // In production, add TransformerEncoderLayer instances
    }

    private void InitializeEmbeddings()
    {
        // Token embedding: maps vocab to hidden dim
        int vocabSize = _tokenizer.VocabularySize;
        _tokenEmbedding = new Tensor<T>(new[] { vocabSize, _hiddenDim });

        // Initialize with random values (in production, load pre-trained weights)
        Random random = new Random(42);
        for (int i = 0; i < _tokenEmbedding.Length; i++)
        {
            _tokenEmbedding[i] = NumOps.FromDouble(random.NextDouble() * 0.02 - 0.01); // Small random values
        }

        // Positional embedding: adds position information
        int maxLength = _tokenizer.MaxLength;
        _positionalEmbedding = new Tensor<T>(new[] { maxLength, _hiddenDim });

        // Sinusoidal positional encoding
        for (int pos = 0; pos < maxLength; pos++)
        {
            for (int i = 0; i < _hiddenDim; i++)
            {
                double angle = pos / Math.Pow(10000.0, (2.0 * i) / _hiddenDim);
                double value = (i % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle);
                _positionalEmbedding[pos, i] = NumOps.FromDouble(value);
            }
        }
    }

    public Tensor<T> Encode(string text)
    {
        return EncodeBatch(new[] { text });
    }

    public Tensor<T> EncodeBatch(string[] texts)
    {
        int batchSize = texts.Length;
        int seqLen = _tokenizer.MaxLength;

        // Tokenize all texts
        var tokenIdsBatch = new int[batchSize][];
        for (int i = 0; i < batchSize; i++)
        {
            tokenIdsBatch[i] = _tokenizer.Tokenize(texts[i]);
        }

        // Create embeddings tensor [batch_size, seq_len, hidden_dim]
        Tensor<T> embeddings = new Tensor<T>(new[] { batchSize, seqLen, _hiddenDim });

        // Embed each token and add positional encoding
        for (int b = 0; b < batchSize; b++)
        {
            for (int pos = 0; pos < seqLen; pos++)
            {
                int tokenId = tokenIdsBatch[b][pos];

                // Look up token embedding
                for (int d = 0; d < _hiddenDim; d++)
                {
                    T tokenEmb = _tokenEmbedding[tokenId, d];
                    T posEmb = _positionalEmbedding[pos, d];
                    embeddings[b, pos, d] = NumOps.Add(tokenEmb, posEmb);
                }
            }
        }

        // Pass through transformer layers (simplified)
        // In production, apply self-attention and feedforward layers
        var output = embeddings;

        // In full implementation:
        // foreach (var layer in _transformerLayers)
        // {
        //     output = layer.Forward(output);
        // }

        return output;
    }
}
```

---

### Step 3: Create Cross-Attention Layer

#### File: `src/NeuralNetworks/Layers/CrossAttentionLayer.cs`
```csharp
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Mathematics;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Cross-attention layer for conditioning on external context.
/// </summary>
/// <typeparam name="T">Numeric type for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Allows image features to attend to text features.
///
/// Self-Attention vs Cross-Attention:
/// - Self-attention: Image attends to itself (what parts of image are related?)
/// - Cross-attention: Image attends to text (what text describes this part?)
///
/// Mechanism:
/// - Query (Q): From image features - "what am I looking for?"
/// - Key (K): From text features - "here's what the text says"
/// - Value (V): From text features - "here's the actual text content"
///
/// Process:
/// 1. Compute attention weights: Q * K^T (how relevant is each text token?)
/// 2. Apply softmax: normalize weights to sum to 1
/// 3. Weighted sum: weights * V (blend text features based on relevance)
///
/// Example:
/// - Generating fur texture in image
/// - Query: "need texture information"
/// - Text: "a fluffy cat with soft fur"
/// - High attention on "fluffy" and "soft fur" tokens
/// - Use those features to guide texture generation
///
/// This is THE key innovation in text-to-image models!
/// </remarks>
public class CrossAttentionLayer<T> : LayerBase<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _embedDim;
    private readonly int _numHeads;
    private readonly int _headDim;

    // Projection weights
    private Tensor<T> _Wq; // Query projection [embed_dim, embed_dim]
    private Tensor<T> _Wk; // Key projection [context_dim, embed_dim]
    private Tensor<T> _Wv; // Value projection [context_dim, embed_dim]
    private Tensor<T> _Wo; // Output projection [embed_dim, embed_dim]

    /// <summary>
    /// Initializes cross-attention layer.
    /// </summary>
    /// <param name="embedDim">
    /// Dimension of image features (query). Typical: 320, 640, 1280.
    /// </param>
    /// <param name="contextDim">
    /// Dimension of text features (key/value). Typical: 768 (CLIP).
    /// </param>
    /// <param name="numHeads">
    /// Number of attention heads. Default: 8.
    /// More heads = more parallel attention patterns.
    /// </param>
    /// <remarks>
    /// <b>For Beginners:</b> Sets up cross-attention.
    ///
    /// Key parameters:
    /// - embedDim: Size of image features
    /// - contextDim: Size of text features (from CLIP)
    /// - numHeads: How many attention patterns to learn
    ///
    /// Multi-head attention learns different types of relationships:
    /// - Head 1: Might focus on objects
    /// - Head 2: Might focus on colors
    /// - Head 3: Might focus on textures
    /// </remarks>
    public CrossAttentionLayer(int embedDim, int contextDim, int numHeads = 8)
    {
        if (embedDim % numHeads != 0)
            throw new ArgumentException($"embedDim ({embedDim}) must be divisible by numHeads ({numHeads})");

        _embedDim = embedDim;
        _numHeads = numHeads;
        _headDim = embedDim / numHeads;

        // Initialize projection weights
        _Wq = InitializeWeight(embedDim, embedDim);
        _Wk = InitializeWeight(contextDim, embedDim);
        _Wv = InitializeWeight(contextDim, embedDim);
        _Wo = InitializeWeight(embedDim, embedDim);
    }

    private Tensor<T> InitializeWeight(int inputDim, int outputDim)
    {
        // Xavier initialization
        double scale = Math.Sqrt(2.0 / (inputDim + outputDim));
        var weight = new Tensor<T>(new[] { inputDim, outputDim });

        Random random = new Random();
        for (int i = 0; i < weight.Length; i++)
        {
            weight[i] = NumOps.FromDouble((random.NextDouble() * 2 - 1) * scale);
        }

        return weight;
    }

    /// <summary>
    /// Forward pass with cross-attention.
    /// </summary>
    /// <param name="input">Image features [batch, seq_len, embed_dim].</param>
    /// <param name="context">Text features [batch, context_len, context_dim].</param>
    /// <returns>Attended features [batch, seq_len, embed_dim].</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Applies cross-attention.
    ///
    /// Steps:
    /// 1. Project image → queries
    /// 2. Project text → keys and values
    /// 3. Compute attention: Q * K^T / sqrt(head_dim)
    /// 4. Softmax to get weights
    /// 5. Weighted sum: weights * V
    /// 6. Concatenate heads and project output
    ///
    /// Shape transformations:
    /// Input: [batch, seq_len, embed_dim]
    /// After attention: [batch, num_heads, seq_len, head_dim]
    /// After concat: [batch, seq_len, embed_dim]
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input, Tensor<T> context)
    {
        int batch = input.Shape[0];
        int seqLen = input.Shape[1];
        int contextLen = context.Shape[1];

        // Project to Q, K, V
        Tensor<T> Q = ProjectQuery(input);       // [batch, seq_len, embed_dim]
        Tensor<T> K = ProjectKey(context);       // [batch, context_len, embed_dim]
        Tensor<T> V = ProjectValue(context);     // [batch, context_len, embed_dim]

        // Reshape for multi-head attention
        // [batch, seq_len, embed_dim] → [batch, num_heads, seq_len, head_dim]
        Q = ReshapeForHeads(Q, batch, seqLen);
        K = ReshapeForHeads(K, batch, contextLen);
        V = ReshapeForHeads(V, batch, contextLen);

        // Compute attention scores: Q * K^T
        // [batch, num_heads, seq_len, head_dim] * [batch, num_heads, head_dim, context_len]
        // → [batch, num_heads, seq_len, context_len]
        Tensor<T> scores = ComputeAttentionScores(Q, K);

        // Scale by sqrt(head_dim)
        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDim));
        scores = scores.Multiply(scale);

        // Apply softmax
        Tensor<T> attentionWeights = Softmax(scores);

        // Apply attention to values: weights * V
        // [batch, num_heads, seq_len, context_len] * [batch, num_heads, context_len, head_dim]
        // → [batch, num_heads, seq_len, head_dim]
        Tensor<T> attended = ApplyAttention(attentionWeights, V);

        // Concatenate heads and project output
        // [batch, num_heads, seq_len, head_dim] → [batch, seq_len, embed_dim]
        Tensor<T> output = ConcatenateHeads(attended, batch, seqLen);
        output = ProjectOutput(output);

        return output;
    }

    private Tensor<T> ProjectQuery(Tensor<T> input)
    {
        // Simplified matrix multiplication
        // In production, use optimized BLAS operations
        return input.MatrixMultiply(_Wq);
    }

    private Tensor<T> ProjectKey(Tensor<T> context)
    {
        return context.MatrixMultiply(_Wk);
    }

    private Tensor<T> ProjectValue(Tensor<T> context)
    {
        return context.MatrixMultiply(_Wv);
    }

    private Tensor<T> ReshapeForHeads(Tensor<T> tensor, int batch, int seqLen)
    {
        // Reshape [batch, seq_len, embed_dim] → [batch, num_heads, seq_len, head_dim]
        var newShape = new[] { batch, _numHeads, seqLen, _headDim };
        return tensor.Reshape(newShape);
    }

    private Tensor<T> ComputeAttentionScores(Tensor<T> Q, Tensor<T> K)
    {
        // Q * K^T
        // Simplified - in production, use batched matrix multiplication
        return Q.MatrixMultiply(K.Transpose());
    }

    private Tensor<T> Softmax(Tensor<T> scores)
    {
        // Apply softmax along last dimension
        // In production, use numerically stable softmax
        var result = new Tensor<T>(scores.Shape);

        // For each batch, head, and query position
        for (int b = 0; b < scores.Shape[0]; b++)
        {
            for (int h = 0; h < scores.Shape[1]; h++)
            {
                for (int i = 0; i < scores.Shape[2]; i++)
                {
                    // Find max for numerical stability
                    T maxScore = NumOps.Zero;
                    for (int j = 0; j < scores.Shape[3]; j++)
                    {
                        T score = scores[b, h, i, j];
                        if (NumOps.GreaterThan(score, maxScore))
                            maxScore = score;
                    }

                    // Compute exp(score - max) and sum
                    T sum = NumOps.Zero;
                    for (int j = 0; j < scores.Shape[3]; j++)
                    {
                        T expScore = NumOps.Exp(NumOps.Subtract(scores[b, h, i, j], maxScore));
                        result[b, h, i, j] = expScore;
                        sum = NumOps.Add(sum, expScore);
                    }

                    // Normalize
                    for (int j = 0; j < scores.Shape[3]; j++)
                    {
                        result[b, h, i, j] = NumOps.Divide(result[b, h, i, j], sum);
                    }
                }
            }
        }

        return result;
    }

    private Tensor<T> ApplyAttention(Tensor<T> weights, Tensor<T> values)
    {
        // weights * values
        return weights.MatrixMultiply(values);
    }

    private Tensor<T> ConcatenateHeads(Tensor<T> tensor, int batch, int seqLen)
    {
        // [batch, num_heads, seq_len, head_dim] → [batch, seq_len, embed_dim]
        var newShape = new[] { batch, seqLen, _embedDim };
        return tensor.Reshape(newShape);
    }

    private Tensor<T> ProjectOutput(Tensor<T> tensor)
    {
        return tensor.MatrixMultiply(_Wo);
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        throw new InvalidOperationException("CrossAttentionLayer requires context. Use Forward(input, context) instead.");
    }

    public override Tensor<T> Backward(Tensor<T> input, Tensor<T> gradients)
    {
        throw new NotImplementedException("Backward pass not implemented in this guide");
    }

    public override int ParameterCount => _Wq.Length + _Wk.Length + _Wv.Length + _Wo.Length;
}
```

---

### Step 4: Implement Latent Diffusion Model with Text Conditioning

#### File: `src/Models/Generative/Diffusion/TextConditionedDiffusion.cs`

```csharp
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Encoders;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Mathematics;

namespace AiDotNet.Models.Generative.Diffusion;

/// <summary>
/// Text-conditioned latent diffusion model.
/// </summary>
/// <typeparam name="T">Numeric type for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Generates images from text prompts.
///
/// This is the architecture behind models like Stable Diffusion.
///
/// Components:
/// 1. Text Encoder (CLIP): Converts prompt to embeddings
/// 2. VAE Encoder: Compresses images to latents (training only)
/// 3. U-Net with Cross-Attention: Denoises latents conditioned on text
/// 4. VAE Decoder: Decompresses latents to images
/// 5. Scheduler: Controls denoising trajectory
///
/// Process:
/// Text: "a cat wearing sunglasses"
///   ↓
/// CLIP: [text embeddings]
///   ↓
/// U-Net denoises random latents, attending to text at each step
///   ↓
/// VAE Decoder: Final image
///
/// Classifier-Free Guidance (CFG):
/// - Generate with AND without text
/// - Emphasize the difference
/// - Makes model follow prompt more strongly
///
/// Formula: output = unconditional + scale * (conditional - unconditional)
/// Typical scale: 7.5
/// </remarks>
public class TextConditionedDiffusion<T> : IDiffusionModel<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IStepScheduler<T> _scheduler;
    private readonly ITextEncoder<T> _textEncoder;
    private readonly IVAE<T>? _vae;
    private readonly INeuralNetwork<T>? _unet;

    private readonly int _latentHeight;
    private readonly int _latentWidth;
    private readonly int _latentChannels;

    public IStepScheduler<T> Scheduler => _scheduler;

    /// <summary>
    /// Initializes text-conditioned diffusion model.
    /// </summary>
    /// <param name="scheduler">Denoising scheduler.</param>
    /// <param name="textEncoder">Text encoder (CLIP).</param>
    /// <param name="vae">VAE for latent space (optional for inference).</param>
    /// <param name="unet">U-Net with cross-attention (must be trained).</param>
    /// <param name="latentHeight">
    /// Height of latent space. Default: 64.
    /// For 512x512 images with 8x compression: 512/8 = 64.
    /// </param>
    /// <param name="latentWidth">Width of latent space. Default: 64.</param>
    /// <param name="latentChannels">Latent channels. Default: 4 (Stable Diffusion standard).</param>
    /// <remarks>
    /// <b>For Beginners:</b> Sets up the complete pipeline.
    ///
    /// Latent space is 8x smaller than image space:
    /// - 512x512 image → 64x64 latent
    /// - 4 channels in latent (vs 3 RGB in image)
    /// - 48x faster generation than pixel-space diffusion!
    /// </remarks>
    public TextConditionedDiffusion(
        IStepScheduler<T> scheduler,
        ITextEncoder<T> textEncoder,
        IVAE<T>? vae = null,
        INeuralNetwork<T>? unet = null,
        int latentHeight = 64,
        int latentWidth = 64,
        int latentChannels = 4)
    {
        _scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));
        _textEncoder = textEncoder ?? throw new ArgumentNullException(nameof(textEncoder));
        _vae = vae;
        _unet = unet;

        _latentHeight = latentHeight;
        _latentWidth = latentWidth;
        _latentChannels = latentChannels;
    }

    /// <summary>
    /// Generates images from text prompts.
    /// </summary>
    /// <param name="prompt">Text description of desired image.</param>
    /// <param name="numInferenceSteps">
    /// Number of denoising steps. Default: 50.
    /// More steps = higher quality but slower.
    /// </param>
    /// <param name="guidanceScale">
    /// Classifier-free guidance scale. Default: 7.5.
    /// Higher = follows prompt more strictly (but less creative).
    /// Range: 1.0 (no guidance) to 20.0 (very strict).
    /// </param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <returns>Generated image tensor.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Main generation method.
    ///
    /// Parameters explained:
    /// - prompt: What you want to see
    /// - numInferenceSteps: Quality vs speed (20=fast, 50=balanced, 100=best)
    /// - guidanceScale: How strictly to follow prompt
    ///   * 1.0: Ignore prompt (random images)
    ///   * 7.5: Balanced (recommended)
    ///   * 15.0: Very literal interpretation
    /// - seed: Set for reproducible results
    ///
    /// Example:
    /// var image = model.Generate(
    ///     prompt: "a cute cat wearing sunglasses, digital art",
    ///     numInferenceSteps: 50,
    ///     guidanceScale: 7.5,
    ///     seed: 42
    /// );
    /// </remarks>
    public Tensor<T> Generate(
        string prompt,
        int numInferenceSteps = 50,
        T guidanceScale = default(T),
        int? seed = null)
    {
        if (_unet == null)
            throw new InvalidOperationException("U-Net not provided. Model must be trained or loaded.");

        // Default guidance scale
        if (NumOps.Equals(guidanceScale, default(T)))
            guidanceScale = NumOps.FromDouble(7.5);

        // Encode text
        Tensor<T> textEmbeddings = _textEncoder.Encode(prompt);

        // For CFG, also encode unconditional (empty) prompt
        Tensor<T> unconditionalEmbeddings = _textEncoder.Encode("");

        // Concatenate for batch processing
        Tensor<T> conditioningEmbeddings = ConcatenateEmbeddings(unconditionalEmbeddings, textEmbeddings);

        // Set seed
        if (seed.HasValue)
            new Random(seed.Value);

        // Initialize latent noise
        int[] latentShape = new[] { 1, _latentChannels, _latentHeight, _latentWidth };
        Tensor<T> latents = _scheduler.InitNoise(latentShape);

        // Get timesteps
        int[] timesteps = _scheduler.GetTimesteps(numInferenceSteps);

        // Denoising loop with classifier-free guidance
        for (int i = 0; i < timesteps.Length; i++)
        {
            int t = timesteps[i];

            // Duplicate latents for batch (unconditional + conditional)
            Tensor<T> latentModelInput = ConcatenateTensors(latents, latents);

            // Predict noise
            // unet.Forward(latents, timestep, text_embeddings)
            // In full implementation, U-Net would use cross-attention with text embeddings
            Tensor<T> noisePred = latentModelInput; // Placeholder

            // Split predictions
            Tensor<T> noisePredUncond = noisePred.Slice(0, 1); // First half
            Tensor<T> noisePredCond = noisePred.Slice(1, 2);   // Second half

            // Apply classifier-free guidance
            // noise = uncond + scale * (cond - uncond)
            Tensor<T> guidedNoise = ApplyClassifierFreeGuidance(
                noisePredUncond,
                noisePredCond,
                guidanceScale
            );

            // Denoise one step
            latents = _scheduler.Step(guidedNoise, t, latents);
        }

        // Decode latents to image
        if (_vae != null)
        {
            return _vae.Decode(latents);
        }
        else
        {
            // Return latents if no VAE (for debugging)
            return latents;
        }
    }

    private Tensor<T> ConcatenateEmbeddings(Tensor<T> uncond, Tensor<T> cond)
    {
        // Stack along batch dimension
        // [1, seq_len, hidden_dim] + [1, seq_len, hidden_dim] → [2, seq_len, hidden_dim]
        int batchSize = 2;
        int seqLen = uncond.Shape[1];
        int hiddenDim = uncond.Shape[2];

        var result = new Tensor<T>(new[] { batchSize, seqLen, hiddenDim });

        // Copy unconditional (batch 0)
        for (int i = 0; i < seqLen; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                result[0, i, j] = uncond[0, i, j];
            }
        }

        // Copy conditional (batch 1)
        for (int i = 0; i < seqLen; i++)
        {
            for (int j = 0; j < hiddenDim; j++)
            {
                result[1, i, j] = cond[0, i, j];
            }
        }

        return result;
    }

    private Tensor<T> ConcatenateTensors(Tensor<T> a, Tensor<T> b)
    {
        // Stack along batch dimension
        var newShape = a.Shape.ToArray();
        newShape[0] *= 2;
        var result = new Tensor<T>(newShape);

        // Copy tensors
        // Simplified - in production, use efficient array copy
        return result;
    }

    private Tensor<T> ApplyClassifierFreeGuidance(
        Tensor<T> uncond,
        Tensor<T> cond,
        T scale)
    {
        // Formula: uncond + scale * (cond - uncond)
        Tensor<T> diff = cond.Subtract(uncond);
        Tensor<T> scaled = diff.Multiply(scale);
        return uncond.Add(scaled);
    }

    // IDiffusionModel implementation
    public Tensor<T> Generate(int numSamples, int numInferenceSteps, int? seed = null)
    {
        // Default to empty prompt
        return Generate("", numInferenceSteps, seed: seed);
    }

    // Other IFullModel methods omitted for brevity
}
```

---

## Testing Strategy

### Test File: `tests/Diffusion/TextConditioningTests.cs`

```csharp
[Fact]
public void Tokenizer_TokenizesSimpleText()
{
    var tokenizer = new SimpleTokenizer();
    var tokens = tokenizer.Tokenize("a cat");

    Assert.NotNull(tokens);
    Assert.True(tokens.Length == tokenizer.MaxLength);
    Assert.Equal(49406, tokens[0]); // START token
}

[Fact]
public void TextEncoder_ProducesCorrectShape()
{
    var encoder = new CLIPTextEncoder<double>();
    var embeddings = encoder.Encode("a cute cat");

    Assert.Equal(3, embeddings.Shape.Length);
    Assert.Equal(1, embeddings.Shape[0]); // Batch
    Assert.Equal(77, embeddings.Shape[1]); // Sequence
    Assert.Equal(768, embeddings.Shape[2]); // Hidden dim
}

[Fact]
public void CrossAttention_ForwardPassRuns()
{
    var layer = new CrossAttentionLayer<double>(
        embedDim: 320,
        contextDim: 768,
        numHeads: 8
    );

    var image = new Tensor<double>(new[] { 1, 64, 320 });
    var text = new Tensor<double>(new[] { 1, 77, 768 });

    var output = layer.Forward(image, text);

    Assert.Equal(image.Shape, output.Shape);
}

[Fact]
public void TextConditionedDiffusion_GeneratesWithPrompt()
{
    var config = new SchedulerConfig<double>();
    var scheduler = new DDIMScheduler<double>(config);
    var textEncoder = new CLIPTextEncoder<double>();

    var model = new TextConditionedDiffusion<double>(
        scheduler,
        textEncoder,
        latentHeight: 8,
        latentWidth: 8
    );

    // Will throw until U-Net is provided
    Assert.Throws<InvalidOperationException>(() =>
        model.Generate("a cat", numInferenceSteps: 10)
    );
}
```

---

## Common Pitfalls

### Pitfall 1: Forgetting Classifier-Free Guidance
Without CFG, model ignores the prompt! Always compute both conditional and unconditional.

### Pitfall 2: Wrong Embedding Dimensions
CLIP outputs 768-dim, U-Net cross-attention must match. Dimension mismatches cause crashes.

### Pitfall 3: Not Padding Tokens
Tokens must be padded to max_length (77). Unpadded sequences break batch processing.

### Pitfall 4: Incorrect Attention Scaling
Must scale by sqrt(head_dim) or attention explodes. Formula: scores / sqrt(head_dim).

---

## Next Steps

1. Implement VAE encoder/decoder (Issue #298)
2. Build U-Net with cross-attention blocks
3. Add positional encodings for timesteps
4. Train or load pre-trained weights
5. Test end-to-end generation pipeline

---

## Resources

- **CLIP Paper**: "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)
- **Latent Diffusion**: "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)
- **Classifier-Free Guidance**: "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2022)
- **Stable Diffusion**: HuggingFace implementation reference

This is cutting-edge generative AI! Take your time understanding cross-attention - it's the key innovation.
