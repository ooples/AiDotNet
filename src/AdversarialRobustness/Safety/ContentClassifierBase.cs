using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AdversarialRobustness.Safety;

/// <summary>
/// Base class for ML-based content classifiers.
/// </summary>
/// <remarks>
/// <para>
/// This abstract class provides common functionality for content classifiers,
/// including threshold-based filtering, category management, and result formatting.
/// Subclasses implement the actual ML model for classification.
/// </para>
/// <para><b>For Beginners:</b> This is a template that makes it easier to build
/// different types of content classifiers. It handles the common tasks like
/// comparing scores to thresholds and formatting results, so you can focus
/// on the actual classification logic in your subclass.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public abstract class ContentClassifierBase<T> : IContentClassifier<T>, IModelSerializer
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The detection threshold for classifying content as harmful.
    /// </summary>
    protected T DetectionThreshold { get; set; }

    /// <summary>
    /// The supported content categories for this classifier.
    /// </summary>
    protected string[] SupportedCategories { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Initializes a new instance of the content classifier.
    /// </summary>
    /// <param name="threshold">The detection threshold (default: 0.5).</param>
    /// <param name="categories">The supported categories.</param>
    protected ContentClassifierBase(double threshold = 0.5, string[]? categories = null)
    {
        DetectionThreshold = NumOps.FromDouble(threshold);
        SupportedCategories = categories ?? DefaultCategories;
    }

    /// <inheritdoc/>
    public abstract ContentClassificationResult<T> Classify(Vector<T> content);

    /// <inheritdoc/>
    public virtual ContentClassificationResult<T> ClassifyText(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return new ContentClassificationResult<T>
            {
                IsHarmful = false,
                OverallConfidence = NumOps.One,
                RecommendedAction = "Allow"
            };
        }

        // Convert text to vector representation
        var vector = TextToVector(text);
        return Classify(vector);
    }

    /// <inheritdoc/>
    public virtual ContentClassificationResult<T>[] ClassifyBatch(Matrix<T> contents)
    {
        if (contents == null)
        {
            throw new ArgumentNullException(nameof(contents));
        }

        var results = new ContentClassificationResult<T>[contents.Rows];
        for (int i = 0; i < contents.Rows; i++)
        {
            results[i] = Classify(contents.GetRow(i));
        }

        return results;
    }

    /// <inheritdoc/>
    public virtual string[] GetSupportedCategories() => SupportedCategories;

    /// <inheritdoc/>
    public abstract bool IsReady();

    /// <inheritdoc/>
    public abstract byte[] Serialize();

    /// <inheritdoc/>
    public abstract void Deserialize(byte[] data);

    /// <inheritdoc/>
    public abstract void SaveModel(string filePath);

    /// <inheritdoc/>
    public abstract void LoadModel(string filePath);

    /// <summary>
    /// Converts text to a vector representation for classification.
    /// </summary>
    /// <param name="text">The text to convert.</param>
    /// <returns>Vector representation of the text.</returns>
    /// <remarks>
    /// Override this method to implement custom text encoding (e.g., tokenization, embeddings).
    /// The default implementation creates a simple character-frequency representation.
    /// </remarks>
    protected virtual Vector<T> TextToVector(string text)
    {
        // Simple default implementation: character frequency vector
        // Subclasses should override with proper tokenization/embedding
        const int VectorSize = 256; // ASCII character space
        var vector = new Vector<T>(VectorSize);

        if (string.IsNullOrEmpty(text))
        {
            return vector;
        }

        // Count character frequencies using LINQ for clarity
        // Group characters by their index position and count occurrences
        var charCounts = text
            .Select(c => Math.Min((int)c, VectorSize - 1))
            .GroupBy(index => index)
            .ToDictionary(g => g.Key, g => g.Count());

        foreach (var kvp in charCounts)
        {
            vector[kvp.Key] = NumOps.FromDouble(kvp.Value);
        }

        // Normalize
        T sum = NumOps.Zero;
        for (int i = 0; i < VectorSize; i++)
        {
            sum = NumOps.Add(sum, vector[i]);
        }

        if (!NumOps.Equals(sum, NumOps.Zero))
        {
            for (int i = 0; i < VectorSize; i++)
            {
                vector[i] = NumOps.Divide(vector[i], sum);
            }
        }

        return vector;
    }

    /// <summary>
    /// Creates a classification result from category scores.
    /// </summary>
    /// <param name="categoryScores">Dictionary of category names to scores.</param>
    /// <returns>Formatted classification result.</returns>
    protected ContentClassificationResult<T> CreateResultFromScores(Dictionary<string, T> categoryScores)
    {
        var result = new ContentClassificationResult<T>
        {
            CategoryScores = categoryScores
        };

        // Find primary category and detected categories
        string primaryCategory = string.Empty;
        T maxScore = NumOps.Zero;
        var detectedCategories = new List<string>();

        foreach (var kvp in categoryScores)
        {
            if (NumOps.GreaterThan(kvp.Value, maxScore))
            {
                maxScore = kvp.Value;
                primaryCategory = kvp.Key;
            }

            if (NumOps.GreaterThan(kvp.Value, DetectionThreshold))
            {
                detectedCategories.Add(kvp.Key);
            }
        }

        result.PrimaryCategory = primaryCategory;
        result.DetectedCategories = detectedCategories.ToArray();
        result.OverallConfidence = maxScore;
        result.IsHarmful = detectedCategories.Count > 0;

        // Determine recommended action
        double maxScoreDouble = NumOps.ToDouble(maxScore);
        if (maxScoreDouble > 0.8)
        {
            result.RecommendedAction = "Block";
        }
        else if (maxScoreDouble > 0.5)
        {
            result.RecommendedAction = "Warn";
        }
        else
        {
            result.RecommendedAction = "Allow";
        }

        return result;
    }

    /// <summary>
    /// The default categories for content classification.
    /// </summary>
    /// <remarks>
    /// Used as a static constant to avoid virtual calls in constructor.
    /// Subclasses can provide their own categories via the constructor parameter.
    /// </remarks>
    protected static readonly string[] DefaultCategories = new[]
    {
        "Safe",
        "Toxic",
        "Violence",
        "HateSpeech",
        "AdultContent",
        "Harassment",
        "SelfHarm",
        "PrivateInformation"
    };
}
