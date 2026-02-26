namespace AiDotNet.VisionLanguage.Interfaces;

/// <summary>
/// Interface for remote sensing vision-language models specializing in satellite and aerial imagery.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Remote sensing VLMs understand satellite imagery, aerial photos, and geospatial data.
/// They support tasks like scene classification, object detection in overhead imagery,
/// change detection, and grounded visual question answering on geospatial data.
/// </para>
/// </remarks>
public interface IRemoteSensingVLM<T> : IGenerativeVisionLanguageModel<T>
{
    /// <summary>
    /// Answers a question about a remote sensing image.
    /// </summary>
    /// <param name="image">Satellite/aerial image tensor in [channels, height, width] format.</param>
    /// <param name="question">Question about the remote sensing image.</param>
    /// <returns>Output tensor of token logits for text generation.</returns>
    Tensor<T> AnswerRemoteSensingQuestion(Tensor<T> image, string question);

    /// <summary>
    /// Gets the supported image resolution bands (e.g., RGB, multispectral).
    /// </summary>
    string SupportedBands { get; }

    /// <summary>
    /// Gets the name of the language model backbone.
    /// </summary>
    string LanguageModelName { get; }
}
