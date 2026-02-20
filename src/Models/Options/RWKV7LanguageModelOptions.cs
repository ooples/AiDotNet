namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the RWKV-7 "Goose" language model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// RWKV-7 is the seventh generation of the RWKV architecture, introducing expressive dynamic
/// state evolution that replaces the fixed exponential decay of previous versions with learnable,
/// data-dependent transition matrices.
/// </para>
/// <para><b>For Beginners:</b> RWKV-7 is a text generation model that processes text in linear time:
///
/// <b>Key Advantages:</b>
/// - Linear time complexity: O(n) vs O(n^2) for Transformers
/// - Constant memory per token during generation
/// - Competitive quality with Transformer models of similar size
///
/// <b>Architecture:</b>
/// 1. Token embedding converts words to vectors
/// 2. N RWKV-7 blocks process the sequence, each with:
///    - Time mixing: WKV-7 kernel with dynamic state evolution
///    - Channel mixing: SiLU-gated feed-forward network
/// 3. RMS normalization for stability
/// 4. LM head projects to vocabulary logits
///
/// <b>Key Innovation (WKV-7):</b>
/// Instead of fixed exponential decay (RWKV-6), the state transition matrices a_t and b_t
/// are learnable and data-dependent:
///   S_t = diag(sigmoid(a_t)) * S_{t-1} + sigmoid(b_t) * outer(k_t, v_t)
///
/// This allows the model to dynamically decide what to remember and forget.
///
/// <b>Typical Model Sizes:</b>
/// - 0.1B: modelDim=768, numLayers=12, numHeads=12
/// - 1.5B: modelDim=2048, numLayers=24, numHeads=32
/// - 7B:   modelDim=4096, numLayers=32, numHeads=64
/// </para>
/// <para>
/// <b>Reference:</b> Peng et al., "RWKV-7 Goose with Expressive Dynamic State Evolution", 2025.
/// </para>
/// </remarks>
public class RWKV7LanguageModelOptions<T> : ModelOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public RWKV7LanguageModelOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public RWKV7LanguageModelOptions(RWKV7LanguageModelOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        VocabSize = other.VocabSize;
        ModelDimension = other.ModelDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        FFNMultiplier = other.FFNMultiplier;
        MaxSequenceLength = other.MaxSequenceLength;
        DropoutRate = other.DropoutRate;
    }

    /// <summary>
    /// Gets or sets the vocabulary size.
    /// </summary>
    /// <value>The vocabulary size, defaulting to 65536 (RWKV-7 standard BPE vocab).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many different tokens (words/subwords) the model knows.
    /// RWKV-7 models typically use a vocabulary of 65536 tokens. Smaller values can be used
    /// for testing or specialized domains.
    /// </para>
    /// </remarks>
    public int VocabSize { get; set; } = 65536;

    /// <summary>
    /// Gets or sets the model dimension (d_model).
    /// </summary>
    /// <value>The model dimension, defaulting to 768.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The width of the hidden representation. This is the main
    /// dimension that flows through the model. Larger values increase capacity but use more
    /// memory and compute. Must be divisible by <see cref="NumHeads"/>.
    ///
    /// Common values: 768 (0.1B), 2048 (1.5B), 4096 (7B).
    /// </para>
    /// </remarks>
    public int ModelDimension { get; set; } = 768;

    /// <summary>
    /// Gets or sets the number of RWKV-7 blocks.
    /// </summary>
    /// <value>The number of layers, defaulting to 12.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The depth of the network. Each layer adds more capacity
    /// for understanding complex patterns. More layers = deeper understanding but slower.
    ///
    /// Common values: 12 (0.1B), 24 (1.5B), 32 (7B).
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of heads per block.
    /// </summary>
    /// <value>The number of heads, defaulting to 12.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each RWKV-7 block maintains a matrix-valued recurrent state
    /// per head. More heads allow the model to track multiple "perspectives" simultaneously.
    /// The head dimension (HeadSize) is ModelDimension / NumHeads, typically 64.
    ///
    /// Must evenly divide <see cref="ModelDimension"/>.
    /// Common values: 12 (0.1B), 32 (1.5B), 64 (7B).
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 12;

    /// <summary>
    /// Gets or sets the FFN expansion multiplier.
    /// </summary>
    /// <value>The FFN multiplier, defaulting to 3.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The channel mixing sub-layer expands the hidden dimension
    /// by this factor before projecting back. 3.5x is the standard for RWKV-7 models.
    /// Higher values increase the capacity of each layer's feed-forward network.
    /// </para>
    /// </remarks>
    public double FFNMultiplier { get; set; } = 3.5;

    /// <summary>
    /// Gets or sets the maximum sequence length.
    /// </summary>
    /// <value>The max sequence length, defaulting to 4096.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The longest text the model can process in one pass during
    /// training. During generation (sequential mode), there is no practical limit since RWKV-7
    /// uses constant memory per token via its recurrent state.
    ///
    /// This mainly affects training: longer sequences use more memory.
    /// </para>
    /// </remarks>
    public int MaxSequenceLength { get; set; } = 4096;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.0 (no dropout, following RWKV convention).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout randomly zeroes some values during training to prevent
    /// overfitting. RWKV models typically use 0.0 dropout, relying on other regularization
    /// techniques. Set higher (0.05-0.1) for small datasets.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.0;
}
