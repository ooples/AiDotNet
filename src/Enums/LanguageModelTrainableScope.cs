namespace AiDotNet.Enums;

/// <summary>Which parts of a multimodal language model a training step may update.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> Large language models are expensive to retrain and easy to damage with a
/// small dataset. Many recipes keep most of the model fixed and train only the pieces that connect it to
/// the new task. This setting chooses how much of the model learns.</para>
/// </remarks>
public enum LanguageModelTrainableScope
{
    /// <summary>Every parameter of the model is trainable.</summary>
    All,

    /// <summary>
    /// Only the word (token) embeddings and the language-model head are trainable; the vision encoder,
    /// projector, transformer layers and positional embeddings are frozen. MGIE's recipe (Fu et al. 2024).
    /// </summary>
    WordEmbeddingsAndHead,

    /// <summary>The whole model is frozen and used only to produce features.</summary>
    None
}
