namespace AiDotNet.Document.Interfaces;

/// <summary>
/// Interface for page segmentation models that identify different regions in document pages.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Page segmentation models divide a document page into semantic regions like
/// text blocks, figures, tables, headers, footers, and captions.
/// </para>
/// <para>
/// <b>For Beginners:</b> When you look at a document page, you can easily identify
/// different sections - titles, paragraphs, images, tables. Page segmentation teaches
/// computers to do the same thing, labeling each region with its type.
///
/// Example usage:
/// <code>
/// var segmenter = new DocBank&lt;float&gt;(architecture);
/// var result = segmenter.SegmentPage(documentImage);
/// foreach (var region in result.Regions)
/// {
///     Console.WriteLine($"Found {region.RegionType} at {region.BoundingBox}");
/// }
/// </code>
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("PageSegmenter")]
public interface IPageSegmenter<T> : IDocumentModel<T>
{
    /// <summary>
    /// Segments a document page into semantic regions.
    /// </summary>
    /// <param name="documentImage">The document page image tensor.</param>
    /// <returns>Segmentation result with labeled regions.</returns>
    PageSegmentationResult<T> SegmentPage(Tensor<T> documentImage);

    /// <summary>
    /// Segments a document page with a custom confidence threshold.
    /// </summary>
    /// <param name="documentImage">The document page image tensor.</param>
    /// <param name="confidenceThreshold">Minimum confidence for region detection (0-1).</param>
    /// <returns>Segmentation result with labeled regions.</returns>
    PageSegmentationResult<T> SegmentPage(Tensor<T> documentImage, double confidenceThreshold);

    /// <summary>
    /// Gets the pixel-level segmentation mask.
    /// </summary>
    /// <param name="documentImage">The document page image tensor.</param>
    /// <returns>Segmentation mask with class indices for each pixel.</returns>
    Tensor<T> GetSegmentationMask(Tensor<T> documentImage);

    /// <summary>
    /// Gets the region types this segmenter can detect.
    /// </summary>
    IReadOnlyList<DocumentRegionType> SupportedRegionTypes { get; }

    /// <summary>
    /// Gets whether this segmenter performs instance segmentation (separate instances of same type).
    /// </summary>
    bool SupportsInstanceSegmentation { get; }
}
