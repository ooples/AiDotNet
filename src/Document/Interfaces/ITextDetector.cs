namespace AiDotNet.Document.Interfaces;

/// <summary>
/// Interface for text detection models that locate text regions in images.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Text detection models find where text appears in an image without reading the text.
/// They output bounding boxes (polygons or rectangles) around text regions.
/// </para>
/// <para>
/// <b>For Beginners:</b> Text detection is the first step in reading text from images.
/// It's like highlighting all the places where text appears, but not actually reading it.
/// After detection, a text recognizer reads the actual characters in each highlighted region.
///
/// Example usage:
/// <code>
/// var detector = new DBNet&lt;float&gt;(architecture);
/// var result = detector.DetectText(documentImage);
/// foreach (var region in result.TextRegions)
/// {
///     Console.WriteLine($"Found text at: {region.BoundingBox}");
/// }
/// </code>
/// </para>
/// </remarks>
public interface ITextDetector<T> : IDocumentModel<T>
{
    /// <summary>
    /// Detects text regions in an image.
    /// </summary>
    /// <param name="image">The input image tensor.</param>
    /// <returns>Detection result with text region locations.</returns>
    TextDetectionResult<T> DetectText(Tensor<T> image);

    /// <summary>
    /// Detects text regions with a custom confidence threshold.
    /// </summary>
    /// <param name="image">The input image tensor.</param>
    /// <param name="confidenceThreshold">Minimum confidence for a detection (0-1).</param>
    /// <returns>Detection result with text region locations.</returns>
    TextDetectionResult<T> DetectText(Tensor<T> image, double confidenceThreshold);

    /// <summary>
    /// Gets the probability map showing text likelihood at each pixel.
    /// </summary>
    /// <param name="image">The input image tensor.</param>
    /// <returns>Probability map tensor with same spatial dimensions as input.</returns>
    Tensor<T> GetProbabilityMap(Tensor<T> image);

    /// <summary>
    /// Gets whether this detector supports rotated text detection.
    /// </summary>
    bool SupportsRotatedText { get; }

    /// <summary>
    /// Gets whether this detector outputs polygon bounding boxes (vs axis-aligned rectangles).
    /// </summary>
    bool SupportsPolygonOutput { get; }

    /// <summary>
    /// Gets the minimum detectable text height in pixels.
    /// </summary>
    int MinTextHeight { get; }
}
