namespace AiDotNet.Enums;

/// <summary>
/// How a text embedding model collapses a sequence of token representations into a single
/// vector for the whole piece of text.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> An embedding model produces one vector per token — one per word-piece
/// of your sentence. To compare two sentences you need a single vector for each, so the token
/// vectors are combined ("pooled"). This chooses how.
/// </para>
/// <para>
/// This was previously nested inside <c>TransformerEmbeddingNetwork&lt;T&gt;</c>. A type nested
/// in a generic class is a distinct type for every type argument, so
/// <c>TransformerEmbeddingNetwork&lt;float&gt;.PoolingStrategy</c> and the <c>double</c> one
/// were unrelated types, and neither could be named from a non-generic options class. Promoting
/// it here makes it usable as configuration, which is what it always described.
/// </para>
/// <para>
/// Not to be confused with <c>AiDotNet.VisionLanguage.Encoders.PoolingStrategy</c>, which pools
/// image patches and offers a different set of choices.
/// </para>
/// </remarks>
public enum EmbeddingPoolingStrategy
{
    /// <summary>
    /// Averages all token representations across the sequence.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The usual choice, and the one most sentence-embedding models
    /// are trained with. Every token contributes equally.</para>
    /// </remarks>
    Mean,

    /// <summary>
    /// Takes the maximum value across all sequence positions for each dimension.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Keeps the strongest signal for each feature rather than the
    /// average, so a single distinctive word can dominate.</para>
    /// </remarks>
    Max,

    /// <summary>
    /// Uses the representation of the first token, typically the [CLS] token.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> BERT-style models prepend a special token whose final
    /// representation is trained to summarise the whole sequence. This uses that.</para>
    /// </remarks>
    ClsToken
}
