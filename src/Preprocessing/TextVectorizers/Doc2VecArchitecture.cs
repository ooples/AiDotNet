namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Specifies the Doc2Vec training architecture.
/// </summary>
public enum Doc2VecArchitecture
{
    /// <summary>
    /// Paragraph Vector - Distributed Memory (PV-DM).
    /// Uses document vector combined with context words to predict target word.
    /// Generally produces better results but slower to train.
    /// </summary>
    PV_DM,

    /// <summary>
    /// Paragraph Vector - Distributed Bag of Words (PV-DBOW).
    /// Uses only document vector to predict words in the document.
    /// Faster training, works well with short documents.
    /// </summary>
    PV_DBOW
}
