namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Specifies the format of pre-trained embeddings files.
/// </summary>
public enum PretrainedFormat
{
    /// <summary>
    /// Auto-detect format from file content.
    /// </summary>
    Auto,

    /// <summary>
    /// Word2Vec text format. First line: vocab_size dim_size.
    /// Following lines: word val1 val2 ... valN
    /// </summary>
    Word2Vec,

    /// <summary>
    /// GloVe text format. Each line: word val1 val2 ... valN
    /// No header line.
    /// </summary>
    GloVe,

    /// <summary>
    /// FastText text format. Same as Word2Vec text format.
    /// </summary>
    FastText,

    /// <summary>
    /// Embeddings loaded from a dictionary (not from file).
    /// </summary>
    Dictionary
}
