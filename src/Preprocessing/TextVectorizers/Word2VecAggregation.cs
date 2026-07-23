namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Specifies how to aggregate word vectors into document vectors.
/// </summary>
public enum Word2VecAggregation
{
    /// <summary>
    /// Average of all word vectors in the document.
    /// </summary>
    Mean,

    /// <summary>
    /// Sum of all word vectors in the document.
    /// </summary>
    Sum,

    /// <summary>
    /// Element-wise maximum across all word vectors.
    /// </summary>
    Max,

    /// <summary>
    /// TF-IDF weighted average of word vectors.
    /// </summary>
    TfidfWeighted
}
