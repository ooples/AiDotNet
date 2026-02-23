namespace AiDotNet.VisionLanguage.Interfaces;

/// <summary>
/// Interface for medical domain vision-language models specializing in biomedical image understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Medical VLMs are trained on biomedical image-text pairs for tasks such as radiology report generation,
/// medical visual question answering, pathology analysis, and clinical decision support.
/// </para>
/// </remarks>
public interface IMedicalVLM<T> : IGenerativeVisionLanguageModel<T>
{
    /// <summary>
    /// Answers a medical question about a biomedical image.
    /// </summary>
    /// <param name="image">Medical image tensor in [channels, height, width] format.</param>
    /// <param name="question">Clinical question about the image.</param>
    /// <returns>Output tensor of token logits for text generation.</returns>
    Tensor<T> AnswerMedicalQuestion(Tensor<T> image, string question);

    /// <summary>
    /// Gets the medical domain this model specializes in (e.g., "Radiology", "Pathology", "General").
    /// </summary>
    string MedicalDomain { get; }

    /// <summary>
    /// Gets the name of the language model backbone.
    /// </summary>
    string LanguageModelName { get; }
}
