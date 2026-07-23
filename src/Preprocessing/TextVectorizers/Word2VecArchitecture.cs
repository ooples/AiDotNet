namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Specifies the Word2Vec training architecture.
/// </summary>
public enum Word2VecArchitecture
{
    /// <summary>
    /// Continuous Bag of Words - predicts target word from context.
    /// Faster training, works well for frequent words.
    /// </summary>
    CBOW,

    /// <summary>
    /// Skip-gram - predicts context words from target word.
    /// Works well with small datasets and rare words.
    /// </summary>
    SkipGram
}
