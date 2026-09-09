using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Shared configuration for text embedding and retrieval models (BGE, ColBERT, SGPT,
/// SPLADE, SimCSE, Instructor, Matryoshka, Word2Vec, GloVe, FastText).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> An embedding model turns a piece of text into a list of numbers so
/// that similar texts end up with similar lists. Search engines use this to find documents
/// that mean the same thing as your query even when they use different words. The settings
/// here are the model's size, and each model's own options class ships the values from its
/// paper.
/// </para>
/// <para>
/// Derived options classes assign their paper's values in their parameterless constructor.
/// See <see cref="ModelHyperparameterOptions"/> for why these properties are non-nullable.
/// </para>
/// <para>
/// <b>Pooling strategy is deliberately absent.</b> Two unrelated <c>EmbeddingPoolingStrategy</c>
/// enums exist in this codebase — one nested inside <c>TransformerEmbeddingNetwork&lt;T&gt;</c>
/// and one in <c>AiDotNet.VisionLanguage.Encoders</c>. Choosing between them, or unifying
/// them, is part of wiring the embedding models rather than of declaring this base, so the
/// property is added then rather than guessed at now.
/// </para>
/// </remarks>
public abstract class EmbeddingModelOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets the number of distinct tokens the model's tokenizer covers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> 30522 is the BERT vocabulary, which many of these models
    /// inherit.</para>
    /// </remarks>
    public int VocabSize { get; set; }

    /// <summary>
    /// Gets or sets the length of the vector the model produces for a piece of text.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The size of the output "fingerprint". 768 is the usual
    /// choice for base-sized models; larger means more nuance and more storage per document.</para>
    /// </remarks>
    public int EmbeddingDimension { get; set; }

    /// <summary>
    /// Gets or sets the longest text, in tokens, the model will encode. Longer inputs are
    /// truncated.
    /// </summary>
    public int MaxSequenceLength { get; set; }

    /// <summary>
    /// Gets or sets the number of stacked transformer blocks.
    /// </summary>
    public int NumLayers { get; set; }

    /// <summary>
    /// Gets or sets the number of attention heads per block.
    /// </summary>
    public int NumHeads { get; set; }

    /// <summary>
    /// Gets or sets the width of the feed-forward sub-layer inside each block.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each block briefly widens its representation to do more
    /// computation, then narrows it again. In BERT-base this is 3072, four times the 768-wide
    /// representation.</para>
    /// </remarks>
    public int FeedForwardDim { get; set; }

    /// <summary>
    /// Throws if a dimension every embedding model requires has been left unset.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative, which means the derived options
    /// class did not assign its paper defaults.
    /// </exception>
    protected void ValidateCore()
    {
        Require(VocabSize, nameof(VocabSize));
        Require(EmbeddingDimension, nameof(EmbeddingDimension));
        Require(MaxSequenceLength, nameof(MaxSequenceLength));
    }
}
