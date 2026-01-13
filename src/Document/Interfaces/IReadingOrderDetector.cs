namespace AiDotNet.Document.Interfaces;

/// <summary>
/// Interface for reading order detection models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IReadingOrderDetector<T> : IDocumentModel<T>
{
    /// <summary>
    /// Detects reading order from a document image.
    /// </summary>
    ReadingOrderResult<T> DetectReadingOrder(Tensor<T> documentImage);

    /// <summary>
    /// Detects reading order from a layout result.
    /// </summary>
    ReadingOrderResult<T> DetectReadingOrder(DocumentLayoutResult<T> layoutResult);
}
