namespace AiDotNet.VisionLanguage.Interfaces;

/// <summary>
/// Interface for video-language models that process video frames for temporal understanding and QA.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Video-language models extend image-based VLMs to handle temporal sequences of video frames.
/// Architectures include frame averaging, spatial-temporal convolution, slow/fast pathways,
/// and long-context approaches for processing hour+ videos.
/// </para>
/// </remarks>
public interface IVideoLanguageModel<T> : IGenerativeVisionLanguageModel<T>
{
    /// <summary>
    /// Generates output from a sequence of video frames, optionally conditioned on a text prompt.
    /// </summary>
    /// <param name="frames">List of frame tensors, each in [channels, height, width] format.</param>
    /// <param name="prompt">Optional text prompt/question about the video.</param>
    /// <returns>Output tensor of token logits for text generation.</returns>
    Tensor<T> GenerateFromVideo(IReadOnlyList<Tensor<T>> frames, string? prompt = null);

    /// <summary>
    /// Gets the maximum number of video frames the model can process.
    /// </summary>
    int MaxFrames { get; }

    /// <summary>
    /// Gets the name of the language model backbone.
    /// </summary>
    string LanguageModelName { get; }
}
