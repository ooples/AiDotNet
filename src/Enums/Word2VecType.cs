namespace AiDotNet.Enums;

/// <summary>
/// Specifies the architecture type for Word2Vec models.
/// </summary>
public enum Word2VecType
{
    /// <summary>
    /// Continuous Bag of Words: Predicts the target word from context words.
    /// Faster to train, better for frequent words.
    /// </summary>
    CBOW,

    /// <summary>
    /// Skip-Gram: Predicts context words from the target word.
    /// Works well with small amount of data, represents rare words well.
    /// </summary>
    SkipGram
}
