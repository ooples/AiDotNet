namespace AiDotNet.VisionLanguage.Interfaces;

/// <summary>
/// Interface for document understanding models that process text-heavy images, documents, charts, and tables.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Document understanding models specialize in extracting and reasoning about textual content
/// from document images, including OCR, layout analysis, table extraction, and document QA.
/// Architectures include:
/// <list type="bullet">
/// <item>LayoutLM: Multimodal pre-training with text, layout, and image</item>
/// <item>Donut/Nougat: OCR-free document understanding via image-to-text generation</item>
/// <item>Pix2Struct: Screenshot parsing pre-training for visual language understanding</item>
/// <item>mPLUG-DocOwl: Modular document understanding with visual abstractor</item>
/// </list>
/// </para>
/// </remarks>
public interface IDocumentUnderstandingModel<T> : IGenerativeVisionLanguageModel<T>
{
    /// <summary>
    /// Extracts text content from a document image.
    /// </summary>
    /// <param name="documentImage">Document image tensor in [channels, height, width] format.</param>
    /// <returns>Tensor of token logits representing extracted text.</returns>
    Tensor<T> ExtractText(Tensor<T> documentImage);

    /// <summary>
    /// Answers a question about a document image.
    /// </summary>
    /// <param name="documentImage">Document image tensor in [channels, height, width] format.</param>
    /// <param name="question">Question about the document content.</param>
    /// <returns>Tensor of token logits for the answer.</returns>
    Tensor<T> AnswerDocumentQuestion(Tensor<T> documentImage, string question);

    /// <summary>
    /// Gets whether this model supports OCR-free document understanding (no external OCR needed).
    /// </summary>
    bool IsOcrFree { get; }
}
