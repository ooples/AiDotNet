using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Shared configuration for sequence and language models (Mamba, RWKV, Jamba, Zamba,
/// Griffin, Hawk, xLSTM, RecurrentGemma and relatives).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These are the "size" settings of a language model — how many
/// distinct tokens it knows, how wide each internal representation is, how many stacked
/// blocks it has, and how far back it can look. You rarely need to change them: each
/// model's own options class already ships the sizes from the paper that introduced it.
/// </para>
/// <para>
/// Derived options classes assign their paper's values in their parameterless
/// constructor. For example:
/// <code>
/// public class MambaOptions : SequenceModelOptions
/// {
///     public MambaOptions()
///     {
///         VocabSize = 50277;      // Mamba-130M
///         ModelDimension = 768;
///         NumLayers = 24;
///     }
/// }
/// </code>
/// </para>
/// <para>
/// Properties are non-nullable and defaulted by the derived class rather than nullable
/// with a <c>GetEffectiveX()</c> fallback. A model's layer count has no runtime-dependent
/// best value the way a batch size does — the right value is the paper's, and it is known
/// statically. See the "Model Hyperparameter Options" section of the development
/// guidelines. Infrastructure configuration (telemetry, profiling, AutoML) keeps the
/// nullable pattern.
/// </para>
/// </remarks>
public abstract class SequenceModelOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets the number of distinct tokens the model's embedding table covers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The size of the model's vocabulary — how many different
    /// words or word-pieces it can represent. Fixed by the tokenizer the paper used, so
    /// changing it means retraining from scratch.</para>
    /// </remarks>
    public int VocabSize { get; set; }

    /// <summary>
    /// Gets or sets the width of the model's hidden representation (often written d_model).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many numbers the model uses to describe each token
    /// internally. Wider means more capacity and more memory.</para>
    /// </remarks>
    public int ModelDimension { get; set; }

    /// <summary>
    /// Gets or sets the number of stacked blocks in the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Depth. Each layer refines what the previous one produced.</para>
    /// </remarks>
    public int NumLayers { get; set; }

    /// <summary>
    /// Gets or sets the number of attention heads, for models that use attention.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Attention heads let the model look at several
    /// relationships at once. Purely recurrent models (Mamba, RWKV) may leave this unset.</para>
    /// </remarks>
    public int NumHeads { get; set; }

    /// <summary>
    /// Gets or sets the width of the recurrent state, for state-space models.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How much the model remembers as it walks through a
    /// sequence. Used by Mamba-family models; attention-only models leave it unset.</para>
    /// </remarks>
    public int StateDimension { get; set; }

    /// <summary>
    /// Gets or sets the longest sequence, in tokens, the model is configured to process.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The context window — how much text the model can consider
    /// at once.</para>
    /// </remarks>
    public int MaxSequenceLength { get; set; }

    /// <summary>
    /// Gets or sets how often an attention block is interleaved among recurrent blocks,
    /// in hybrid architectures such as Jamba, Samba and Zamba.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Hybrid models mix two kinds of block. A value of 6 means
    /// every sixth block uses attention and the rest are recurrent. Ignored by
    /// non-hybrid models.</para>
    /// </remarks>
    public int AttentionInterval { get; set; }

    /// <summary>
    /// Gets or sets the inner-projection expansion factor used by state-space blocks.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Mamba-style blocks temporarily widen their representation
    /// before narrowing it again; this is the multiplier. The Mamba paper uses 2.</para>
    /// </remarks>
    public int ExpandFactor { get; set; }

    /// <summary>
    /// Gets or sets the feed-forward width as a multiple of <see cref="ModelDimension"/>,
    /// for models that express it as a ratio (RWKV-7 uses 3.5).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How much wider the model's internal "thinking" layer is
    /// than its representation width.</para>
    /// </remarks>
    public double FfnMultiplier { get; set; }

    /// <summary>
    /// Throws if a dimension this model family requires has been left unset.
    /// </summary>
    /// <param name="requiresHeads">Whether this model uses attention heads.</param>
    /// <param name="requiresState">Whether this model uses a recurrent state dimension.</param>
    /// <param name="requiresVocabulary">
    /// Whether this model's sequence is made of tokens drawn from a vocabulary. False for the
    /// vision members of this family, whose sequence is made of image patches.
    /// </param>
    /// <param name="requiresSequenceLength">
    /// Whether this model has a configured maximum sequence length. False when the length is
    /// derived from the input rather than declared — a patch sequence, for instance, is as long
    /// as the image size and patch size make it.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative, which means the derived options
    /// class did not assign its paper defaults.
    /// </exception>
    /// <remarks>
    /// <para>
    /// Fails loudly rather than silently constructing a zero-width model. A dimension of
    /// zero produces layers that appear to build and then misbehave far from the cause.
    /// </para>
    /// <para>
    /// <b>A base may only require what every member of the family has.</b> Only
    /// <see cref="ModelDimension"/> and <see cref="NumLayers"/> are common to every sequence
    /// model; everything else is asked for by the leaves that actually use it. Requiring more
    /// than that here makes a model throw at its own published defaults, with the user having
    /// configured nothing.
    /// </para>
    /// </remarks>
    protected void ValidateCore(
        bool requiresHeads,
        bool requiresState,
        bool requiresVocabulary = true,
        bool requiresSequenceLength = true)
    {
        Require(ModelDimension, nameof(ModelDimension));
        Require(NumLayers, nameof(NumLayers));
        if (requiresVocabulary) Require(VocabSize, nameof(VocabSize));
        if (requiresSequenceLength) Require(MaxSequenceLength, nameof(MaxSequenceLength));
        if (requiresHeads) Require(NumHeads, nameof(NumHeads));
        if (requiresState) Require(StateDimension, nameof(StateDimension));
    }
}
