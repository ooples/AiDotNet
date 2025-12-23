using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AdversarialRobustness.Safety;

/// <summary>
/// Defines the interface for ML-based content classification.
/// </summary>
/// <remarks>
/// <para>
/// Content classifiers provide machine learning-based analysis of content to detect
/// harmful, toxic, or inappropriate material. Unlike regex-based pattern matching,
/// ML classifiers can understand semantic meaning and context.
/// </para>
/// <para><b>For Beginners:</b> Think of this as an AI-powered content moderator that
/// can understand the meaning of text, not just look for specific keywords. It can detect
/// subtle forms of harmful content that simple pattern matching would miss.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public interface IContentClassifier<T>
{
    /// <summary>
    /// Classifies content and returns the classification result.
    /// </summary>
    /// <param name="content">The content to classify as a vector representation.</param>
    /// <returns>The classification result with category predictions and confidence scores.</returns>
    ContentClassificationResult<T> Classify(Vector<T> content);

    /// <summary>
    /// Classifies content provided as text.
    /// </summary>
    /// <param name="text">The text content to classify.</param>
    /// <returns>The classification result with category predictions and confidence scores.</returns>
    ContentClassificationResult<T> ClassifyText(string text);

    /// <summary>
    /// Classifies a batch of content items.
    /// </summary>
    /// <param name="contents">Matrix where each row is a content item to classify.</param>
    /// <returns>Array of classification results, one per input row.</returns>
    ContentClassificationResult<T>[] ClassifyBatch(Matrix<T> contents);

    /// <summary>
    /// Gets the list of content categories this classifier can detect.
    /// </summary>
    /// <returns>Array of category names supported by this classifier.</returns>
    string[] GetSupportedCategories();

    /// <summary>
    /// Checks if the classifier is ready to make predictions.
    /// </summary>
    /// <returns>True if the model is loaded and ready, false otherwise.</returns>
    bool IsReady();
}

/// <summary>
/// Result of content classification by an ML model.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class ContentClassificationResult<T>
{
    /// <summary>
    /// Gets or sets whether any harmful content was detected.
    /// </summary>
    public bool IsHarmful { get; set; }

    /// <summary>
    /// Gets or sets the overall confidence score for the classification.
    /// </summary>
    public T OverallConfidence { get; set; } = default!;

    /// <summary>
    /// Gets or sets the primary detected category (the one with highest score).
    /// </summary>
    public string PrimaryCategory { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the scores for each category.
    /// </summary>
    public Dictionary<string, T> CategoryScores { get; set; } = new Dictionary<string, T>();

    /// <summary>
    /// Gets or sets the list of categories that exceed the detection threshold.
    /// </summary>
    public string[] DetectedCategories { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets the recommended action based on classification.
    /// </summary>
    public string RecommendedAction { get; set; } = "Allow";

    /// <summary>
    /// Gets or sets any explanation or reasoning for the classification.
    /// </summary>
    public string Explanation { get; set; } = string.Empty;
}
