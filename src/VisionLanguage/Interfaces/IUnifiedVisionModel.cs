namespace AiDotNet.VisionLanguage.Interfaces;

/// <summary>
/// Interface for unified vision models that can both understand and generate images.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Unified models combine visual understanding (captioning, VQA) and visual generation
/// (image synthesis) in a single architecture. Approaches include:
/// <list type="bullet">
/// <item>Chameleon/Show-o: discrete visual tokens for unified autoregressive generation</item>
/// <item>Janus: decoupled visual encoding for understanding vs. generation</item>
/// <item>Transfusion: mixed autoregressive + diffusion loss in one transformer</item>
/// </list>
/// </para>
/// </remarks>
public interface IUnifiedVisionModel<T> : IGenerativeVisionLanguageModel<T>
{
    /// <summary>
    /// Generates an image tensor from a text description.
    /// </summary>
    /// <param name="textDescription">Text prompt describing the desired image.</param>
    /// <returns>Generated image tensor in [channels, height, width] format.</returns>
    Tensor<T> GenerateImage(string textDescription);

    /// <summary>
    /// Gets whether this model supports image generation (not just understanding).
    /// </summary>
    bool SupportsGeneration { get; }
}
