namespace AiDotNet.Document.Interfaces;

/// <summary>
/// Interface for OCR (Optical Character Recognition) models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// OCR models convert images containing text into machine-readable text strings,
/// along with position information for each text element.
/// </para>
/// <para>
/// <b>For Beginners:</b> OCR is like teaching a computer to read. Given an image of text,
/// the model outputs the actual text content and where each word/character is located.
///
/// Example usage:
/// <code>
/// var result = ocrModel.RecognizeText(documentImage);
/// Console.WriteLine($"Full text: {result.FullText}");
/// foreach (var word in result.Words)
/// {
///     Console.WriteLine($"'{word.Text}' at position ({word.BoundingBox})");
/// }
/// </code>
/// </para>
/// </remarks>
public interface IOCRModel<T> : IDocumentModel<T>
{
    /// <summary>
    /// Performs full OCR on a document image.
    /// </summary>
    /// <param name="documentImage">The document image tensor.</param>
    /// <returns>OCR result with text, positions, and confidence scores.</returns>
    OCRResult<T> RecognizeText(Tensor<T> documentImage);

    /// <summary>
    /// Performs OCR on a specific region of the document.
    /// </summary>
    /// <param name="documentImage">The document image tensor.</param>
    /// <param name="region">The region to process as normalized coordinates [x1, y1, x2, y2] where values are 0-1.</param>
    /// <returns>OCR result for the specified region.</returns>
    OCRResult<T> RecognizeTextInRegion(Tensor<T> documentImage, Vector<T> region);

    /// <summary>
    /// Gets the languages supported by this OCR model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Languages are specified using ISO 639-1 codes (e.g., "en", "zh", "ja").
    /// Some models support multiple languages simultaneously.
    /// </para>
    /// </remarks>
    IReadOnlyList<string> SupportedLanguages { get; }

    /// <summary>
    /// Gets whether this is an OCR-free model (end-to-end pixel-to-text).
    /// </summary>
    /// <remarks>
    /// <para>
    /// OCR-free models like Donut directly convert pixels to text without explicit
    /// text detection or recognition stages. Traditional OCR has separate stages.
    /// </para>
    /// </remarks>
    bool IsOCRFree { get; }
}
