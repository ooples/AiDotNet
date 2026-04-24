using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// Base class for text conditioning modules used in diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Text conditioning modules encode text prompts into embedding tensors that guide the
/// diffusion model's generation process. This base class provides common functionality
/// for CLIP, T5, and other text encoders used in diffusion models.
/// </para>
/// <para>
/// <b>For Beginners:</b> A text conditioning module is a "translator" that converts
/// your text prompt into numbers the diffusion model can understand.
///
/// When you type "a cat sitting on a couch", this module:
/// 1. Breaks the text into tokens (word pieces)
/// 2. Converts tokens to IDs using a vocabulary
/// 3. Processes the IDs through a transformer encoder
/// 4. Outputs embedding vectors that represent the meaning of each token
///
/// These embeddings then guide the diffusion model to generate images
/// matching your description.
/// </para>
/// </remarks>
public abstract class TextConditioningBase<T> : IConditioningModule<T>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Hardware-accelerated engine for vector/tensor operations.
    /// </summary>
    protected static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Random number generator for initialization.
    /// </summary>
    protected readonly Random Rng;

    /// <summary>
    /// Token embedding weights [vocabSize, embeddingDim].
    /// </summary>
    protected Vector<T> TokenEmbeddings;

    /// <summary>
    /// Position embedding weights [maxSeqLen, embeddingDim].
    /// </summary>
    protected Vector<T> PositionEmbeddings;

    /// <summary>
    /// Transformer layer weights (flattened).
    /// </summary>
    protected Vector<T> TransformerWeights;

    /// <summary>
    /// Final layer normalization weights.
    /// </summary>
    protected Vector<T> FinalLayerNormWeights;

    /// <summary>
    /// Final layer normalization bias.
    /// </summary>
    protected Vector<T> FinalLayerNormBias;

    /// <summary>
    /// Vocabulary size for the tokenizer.
    /// </summary>
    protected readonly int VocabSize;

    /// <summary>
    /// Hidden dimension of the transformer.
    /// </summary>
    protected readonly int HiddenSize;

    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    protected readonly int NumLayers;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    protected readonly int NumHeads;

    /// <inheritdoc />
    public int EmbeddingDimension { get; }

    /// <inheritdoc />
    public ConditioningType ConditioningType => ConditioningType.Text;

    /// <inheritdoc />
    public abstract bool ProducesPooledOutput { get; }

    /// <inheritdoc />
    public int MaxSequenceLength { get; }

    /// <summary>
    /// Initializes a new instance of the TextConditioningBase class.
    /// </summary>
    /// <param name="vocabSize">The vocabulary size.</param>
    /// <param name="embeddingDimension">The output embedding dimension.</param>
    /// <param name="hiddenSize">The hidden dimension of the transformer.</param>
    /// <param name="numLayers">The number of transformer layers.</param>
    /// <param name="numHeads">The number of attention heads.</param>
    /// <param name="maxSequenceLength">The maximum sequence length.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    protected TextConditioningBase(
        int vocabSize,
        int embeddingDimension,
        int hiddenSize,
        int numLayers,
        int numHeads,
        int maxSequenceLength,
        int? seed = null,
        bool skipBaseWeightAllocation = false)
    {
        VocabSize = vocabSize;
        EmbeddingDimension = embeddingDimension;
        HiddenSize = hiddenSize;
        NumLayers = numLayers;
        NumHeads = numHeads;
        MaxSequenceLength = maxSequenceLength;
        Rng = seed.HasValue ? Tensors.Helpers.RandomHelper.CreateSeededRandom(seed.Value) : Tensors.Helpers.RandomHelper.CreateSecureRandom();

        // Initialize weights with small random values
        TokenEmbeddings = InitializeWeights(vocabSize * hiddenSize);
        PositionEmbeddings = InitializeWeights(maxSequenceLength * hiddenSize);

        // Subclasses that own per-layer weight storage (and specifically
        // subclasses whose variant weights don't fit in a single Vector<T>
        // — e.g., T5-XXL has ~4.6B parameters, ~17× the int32 Vector cap)
        // pass skipBaseWeightAllocation:true and manage their own
        // per-layer rent-and-return via TensorAllocator. Everything else
        // goes through the default path below: a single contiguous
        // Vector<T> sized by the generic `12 * H^2 + 4 * H` per-layer
        // budget, which works for all CLIP/SigLIP/Gemma/Qwen/ChatGLM
        // variants in this codebase.
        if (skipBaseWeightAllocation)
        {
            // Empty placeholder — never read by the default forward
            // path when subclasses opt out of the flat layout.
            TransformerWeights = new Vector<T>(0);
        }
        else
        {
            // Use long arithmetic to compute the requested size, then
            // detect int32 overflow before handing it to Vector<T>
            // (whose ctor takes an int). The silent-wrap behavior of
            // the old code masked T5-XXL-style overflows as a wildly
            // undersized buffer that only blew up downstream inside
            // SliceMatrixTensor (issue #1189).
            long weightsPerLayer = 12L * hiddenSize * hiddenSize + 4L * hiddenSize;
            long totalWeights = (long)numLayers * weightsPerLayer;
            if (totalWeights > int.MaxValue)
            {
                throw new InvalidOperationException(
                    $"TextConditioningBase: requested transformer weight count " +
                    $"({totalWeights:N0} elements) exceeds the int32 Vector<T> " +
                    $"capacity ({int.MaxValue:N0}). This typically means the " +
                    $"variant is T5-XXL (or similarly large). Subclasses that " +
                    $"need to support variants at this scale must pass " +
                    $"skipBaseWeightAllocation:true to this constructor and " +
                    $"manage per-layer weight storage via TensorAllocator " +
                    $"(see T5TextConditioner for the reference pattern).");
            }
            TransformerWeights = InitializeWeights((int)totalWeights);
        }

        FinalLayerNormWeights = new Vector<T>(hiddenSize);
        FinalLayerNormBias = new Vector<T>(hiddenSize);
        for (int i = 0; i < hiddenSize; i++)
        {
            FinalLayerNormWeights[i] = NumOps.One;
            FinalLayerNormBias[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Initializes a weight vector with small random values (Xavier initialization).
    /// </summary>
    protected Vector<T> InitializeWeights(int size)
    {
        var weights = new Vector<T>(size);
        double stddev = Math.Sqrt(2.0 / (size + 1));
        for (int i = 0; i < size; i++)
        {
            double u1 = 1.0 - Rng.NextDouble();
            double u2 = 1.0 - Rng.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            weights[i] = NumOps.FromDouble(normal * stddev);
        }
        return weights;
    }

    /// <inheritdoc />
    public abstract Tensor<T> Encode(Tensor<T> input);

    /// <inheritdoc />
    public abstract Tensor<T> EncodeText(Tensor<T> tokenIds, Tensor<T>? attentionMask = null);

    /// <inheritdoc />
    public abstract Tensor<T> GetPooledEmbedding(Tensor<T> sequenceEmbeddings);

    /// <inheritdoc />
    public abstract Tensor<T> GetUnconditionalEmbedding(int batchSize);

    /// <inheritdoc />
    public abstract Tensor<T> Tokenize(string text);

    /// <inheritdoc />
    public abstract Tensor<T> TokenizeBatch(string[] texts);

    /// <summary>
    /// Applies layer normalization to a vector.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <param name="gamma">Scale parameter (weights).</param>
    /// <param name="beta">Shift parameter (bias).</param>
    /// <param name="dim">The dimension to normalize over.</param>
    /// <returns>The normalized vector.</returns>
    protected Vector<T> LayerNorm(Vector<T> input, Vector<T> gamma, Vector<T> beta, int dim)
    {
        var result = new Vector<T>(input.Length);
        int numVectors = input.Length / dim;

        for (int v = 0; v < numVectors; v++)
        {
            int offset = v * dim;

            // Compute mean
            T sum = NumOps.Zero;
            for (int i = 0; i < dim; i++)
                sum = NumOps.Add(sum, input[offset + i]);
            T mean = NumOps.Divide(sum, NumOps.FromDouble(dim));

            // Compute variance
            T varSum = NumOps.Zero;
            for (int i = 0; i < dim; i++)
            {
                T diff = NumOps.Subtract(input[offset + i], mean);
                varSum = NumOps.Add(varSum, NumOps.Multiply(diff, diff));
            }
            T variance = NumOps.Divide(varSum, NumOps.FromDouble(dim));
            T stddev = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-5)));

            // Normalize, scale, and shift
            for (int i = 0; i < dim; i++)
            {
                T normalized = NumOps.Divide(NumOps.Subtract(input[offset + i], mean), stddev);
                result[offset + i] = NumOps.Add(NumOps.Multiply(gamma[i], normalized), beta[i]);
            }
        }

        return result;
    }

    /// <summary>
    /// Simple tokenization: maps characters to integer IDs (placeholder for real tokenizer).
    /// </summary>
    /// <param name="text">The text to tokenize.</param>
    /// <param name="maxLength">Maximum number of tokens.</param>
    /// <returns>Array of token IDs padded/truncated to maxLength.</returns>
    protected int[] SimpleTokenize(string text, int maxLength)
    {
        var tokens = new int[maxLength];

        // BOS token
        tokens[0] = 1;

        // Simple character-level tokenization as placeholder
        int pos = 1;
        foreach (char c in text)
        {
            if (pos >= maxLength - 1)
                break;
            // Map characters to token range [2, vocabSize-2]
            tokens[pos] = (c % (VocabSize - 3)) + 2;
            pos++;
        }

        // EOS token
        if (pos < maxLength)
            tokens[pos] = VocabSize - 1;

        // Remaining positions are already 0 (padding)
        return tokens;
    }
}
