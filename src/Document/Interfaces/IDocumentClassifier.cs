namespace AiDotNet.Document.Interfaces;

/// <summary>
/// Interface for document classification models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Document classification models categorize documents into predefined classes
/// such as invoices, forms, letters, scientific papers, etc.
/// </para>
/// <para>
/// <b>For Beginners:</b> Document classification is like sorting mail into different piles.
/// The model looks at a document and decides what type it is. This is useful for:
/// - Organizing scanned documents
/// - Routing documents to appropriate processing pipelines
/// - Quality control in document processing
///
/// Example usage:
/// <code>
/// var classifier = new DocumentTypeClassifier&lt;float&gt;(architecture);
/// var result = classifier.ClassifyDocument(documentImage);
/// Console.WriteLine($"Document type: {result.PredictedCategory}");
/// </code>
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("DocumentClassifier")]
public interface IDocumentClassifier<T> : IDocumentModel<T>
{
    /// <summary>
    /// Classifies a document image into predefined categories.
    /// </summary>
    /// <param name="documentImage">The document image tensor.</param>
    /// <returns>Classification result with predicted category and confidence.</returns>
    DocumentClassificationResult<T> ClassifyDocument(Tensor<T> documentImage);

    /// <summary>
    /// Classifies a document and returns top-K predictions.
    /// </summary>
    /// <param name="documentImage">The document image tensor.</param>
    /// <param name="topK">Number of top predictions to return.</param>
    /// <returns>Classification result with top-K predictions.</returns>
    DocumentClassificationResult<T> ClassifyDocument(Tensor<T> documentImage, int topK);

    /// <summary>
    /// Gets the available classification categories for this model.
    /// </summary>
    IReadOnlyList<string> AvailableCategories { get; }
}

/// <summary>
/// Result of document classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DocumentClassificationResult<T>
{
    /// <summary>
    /// Gets or sets the predicted document category.
    /// </summary>
    public string PredictedCategory { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the confidence score as generic type.
    /// </summary>
    public T Confidence { get; set; } = default!;

    /// <summary>
    /// Gets or sets the confidence score as double.
    /// </summary>
    public double ConfidenceValue { get; set; }

    /// <summary>
    /// Gets or sets the top-K predictions with their scores.
    /// </summary>
    public IList<(string Category, double Score)> TopPredictions { get; set; } = [];

    /// <summary>
    /// Gets or sets the processing time in milliseconds.
    /// </summary>
    public double ProcessingTimeMs { get; set; }
}
