namespace AiDotNet.Document.Interfaces;

/// <summary>
/// Interface for document layout detection models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Layout detection identifies and localizes different structural elements in a document,
/// such as text blocks, tables, figures, headers, and footers.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of layout detection as drawing boxes around different parts
/// of a document and labeling what each part is (title, paragraph, table, etc.).
/// This helps computers understand the structure of a document just like humans do.
///
/// Example usage:
/// <code>
/// var result = layoutDetector.DetectLayout(documentImage);
/// foreach (var region in result.Regions)
/// {
///     Console.WriteLine($"Found {region.ElementType} at ({region.BoundingBox})");
/// }
/// </code>
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("LayoutDetector")]
public interface ILayoutDetector<T> : IDocumentModel<T>
{
    /// <summary>
    /// Detects layout regions in a document image.
    /// </summary>
    /// <param name="documentImage">The document image tensor [batch, channels, height, width].</param>
    /// <returns>Layout detection result with regions and their types.</returns>
    DocumentLayoutResult<T> DetectLayout(Tensor<T> documentImage);

    /// <summary>
    /// Detects layout regions with a specified confidence threshold.
    /// </summary>
    /// <param name="documentImage">The document image tensor.</param>
    /// <param name="confidenceThreshold">Minimum confidence for detected regions (0.0 to 1.0).</param>
    /// <returns>Filtered layout detection result.</returns>
    /// <remarks>
    /// <para>
    /// Higher thresholds return fewer but more confident detections.
    /// Lower thresholds return more detections but may include false positives.
    /// </para>
    /// </remarks>
    DocumentLayoutResult<T> DetectLayout(Tensor<T> documentImage, double confidenceThreshold);

    /// <summary>
    /// Gets the layout element types this detector can identify.
    /// </summary>
    IReadOnlyList<LayoutElementType> SupportedElementTypes { get; }
}
