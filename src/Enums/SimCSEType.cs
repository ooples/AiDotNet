namespace AiDotNet.Enums;

/// <summary>
/// Defines the training paradigms for SimCSE (Simple Contrastive Learning of Sentence Embeddings).
/// </summary>
/// <remarks>
/// <para>
/// SimCSE supports two primary modes:
/// <list type="bullet">
/// <item><b>Unsupervised:</b> Uses dropout masks on identical sentence pairs as a minimal data augmentation.</item>
/// <item><b>Supervised:</b> Uses labeled entailment and contradiction pairs from datasets like SNLI or MultiNLI.</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this as the "learning style" of the model. 
/// - <b>Unsupervised</b> is like a student learning by comparing different versions of the same book.
/// - <b>Supervised</b> is like a student learning from a teacher who provides "true" and "false" examples.
/// </para>
/// </remarks>
public enum SimCSEType
{
    /// <summary>
    /// Unsupervised learning using dropout as noise.
    /// </summary>
    Unsupervised,

    /// <summary>
    /// Supervised learning using Natural Language Inference (NLI) datasets.
    /// </summary>
    Supervised
}
