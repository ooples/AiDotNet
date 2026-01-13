using AiDotNet.Interfaces;

namespace AiDotNet.Document.Interfaces;

/// <summary>
/// Base interface for all document AI models in AiDotNet.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This interface extends <see cref="IFullModel{T, TInput, TOutput}"/> to provide the core contract
/// for document AI models, inheriting standard methods for training, inference, model persistence,
/// and gradient computation.
/// </para>
/// <para>
/// <b>For Beginners:</b> A document AI model processes document images (scanned pages, PDFs, photos of text)
/// to extract information, understand layout, or answer questions.
///
/// Key concepts:
/// - Document images have shape [batch, channels, height, width]
/// - Models can run in Native mode (pure C#) or ONNX mode (optimized runtime)
/// - All models support both training and inference
/// - Many document models combine vision and language understanding
///
/// Example usage:
/// <code>
/// var model = new LayoutLMv3&lt;double&gt;(architecture);
/// var layout = model.DetectLayout(documentImage);
/// var text = model.ExtractText(documentImage);
/// </code>
/// </para>
/// </remarks>
public interface IDocumentModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    #region Document-Specific Properties

    /// <summary>
    /// Gets the expected input image size (assumes square images).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Common values: 224 (ViT base), 384, 448, 512, 768, 1024.
    /// Input images will be resized to [ImageSize x ImageSize] before processing.
    /// </para>
    /// </remarks>
    int ExpectedImageSize { get; }

    /// <summary>
    /// Gets the maximum sequence length for text processing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For layout-aware models, this is the maximum number of text tokens.
    /// Common values: 512, 1024, 2048.
    /// </para>
    /// </remarks>
    int MaxSequenceLength { get; }

    /// <summary>
    /// Gets whether this model requires OCR preprocessing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// OCR-free models (Donut, Pix2Struct) return false - they process raw pixels directly.
    /// Layout-aware models (LayoutLM) return true - they need text and bounding boxes from OCR.
    /// </para>
    /// </remarks>
    bool RequiresOCR { get; }

    /// <summary>
    /// Gets the supported document types for this model.
    /// </summary>
    DocumentType SupportedDocumentTypes { get; }

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, the model uses pre-trained ONNX weights for fast inference.
    /// When false, the model uses native layers and can be trained.
    /// </para>
    /// </remarks>
    bool IsOnnxMode { get; }

    #endregion

    #region Document-Specific Methods

    /// <summary>
    /// Processes a document image and returns encoded features.
    /// </summary>
    /// <param name="documentImage">The document image tensor [batch, channels, height, width] or [channels, height, width].</param>
    /// <returns>Encoded document features suitable for downstream tasks.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method converts a document image into a numerical representation
    /// (feature vector) that captures the document's content and structure.
    /// These features can then be used for tasks like classification, QA, or information extraction.
    /// </para>
    /// </remarks>
    Tensor<T> EncodeDocument(Tensor<T> documentImage);

    /// <summary>
    /// Validates that an input tensor has the correct shape for this model.
    /// </summary>
    /// <param name="documentImage">The tensor to validate.</param>
    /// <exception cref="ArgumentException">Thrown if the tensor shape is invalid.</exception>
    void ValidateInputShape(Tensor<T> documentImage);

    /// <summary>
    /// Gets a summary of the model architecture.
    /// </summary>
    /// <returns>A string describing the model's architecture, parameters, and capabilities.</returns>
    string GetModelSummary();

    #endregion
}
