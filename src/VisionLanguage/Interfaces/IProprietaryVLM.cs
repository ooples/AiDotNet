namespace AiDotNet.VisionLanguage.Interfaces;

/// <summary>
/// Interface for reference implementations of proprietary VLM architectures.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Proprietary VLMs represent state-of-the-art commercial models from major AI labs.
/// These reference implementations approximate their published architectures for
/// understanding the design space and benchmarking open alternatives.
/// </para>
/// </remarks>
public interface IProprietaryVLM<T> : IGenerativeVisionLanguageModel<T>
{
    /// <summary>
    /// Generates output from an image with a text prompt in chat-style interaction.
    /// </summary>
    /// <param name="image">Input image tensor in [channels, height, width] format.</param>
    /// <param name="prompt">Text prompt or question about the image.</param>
    /// <returns>Output tensor of token logits for text generation.</returns>
    Tensor<T> Chat(Tensor<T> image, string prompt);

    /// <summary>
    /// Gets the name of the proprietary model provider.
    /// </summary>
    string Provider { get; }

    /// <summary>
    /// Gets the name of the language model backbone.
    /// </summary>
    string LanguageModelName { get; }
}
