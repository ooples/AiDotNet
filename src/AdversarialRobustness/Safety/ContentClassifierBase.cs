using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
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
public abstract partial class ContentClassifierBase<T> : IContentClassifier<T>, IModelSerializer, IModelShape
{
    // --- declared state (ModelStateRegistry) ---
    // Identical in every model base because these bases are siblings over the same interfaces rather
    // than one hierarchy; the logic itself lives once in ModelStateRegistry/ModelStateEnvelope.

    /// <summary>State that is not a parameter vector, declared once and persisted by this base.</summary>
    private readonly AiDotNet.Models.ModelStateRegistry<T> _declaredState = new();
    private bool _declaredStateRegistered;

    /// <summary>
    /// Declare state here that the parameter vector does not carry -- a retained training set,
    /// fitted knots, kernel centres, an ensemble's children. Both halves of the payload are driven
    /// by the declaration, so they cannot drift.
    /// </summary>
    /// <param name="state">The registry to declare into.</param>
    protected virtual void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
    }
    /// <summary>Generated state declarations for fields declared across this model's hierarchy.</summary>
    /// <param name="state">The registry to declare into.</param>
    /// <remarks>
    /// Emitted by ModelStateGenerator into the partial model, so a model author declares nothing. The
    /// hand-written <c>RegisterState</c> beside it exists only for state the classifier genuinely
    /// cannot place; anything it CAN place belongs here, where it cannot be forgotten.
    /// </remarks>
    protected virtual void RegisterGeneratedState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        RegisterGeneratedStateCore(state);
    }

    /// <summary>The declared state, registered once and lazily so it runs after the constructor.</summary>
    protected AiDotNet.Models.ModelStateRegistry<T> DeclaredState
    {
        get
        {
            if (!_declaredStateRegistered)
            {
                _declaredStateRegistered = true;
                RegisterGeneratedState(_declaredState);
                RegisterState(_declaredState);
            }
            return _declaredState;
        }
    }
    /// <summary>
    /// Gets the hardware-accelerated computation engine for vectorized operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

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
    public virtual int[] GetInputShape()
    {
        // Subclasses should override with the actual feature dimension
        return Array.Empty<int>();
    }

    /// <inheritdoc/>
    public virtual int[] GetOutputShape()
    {
        // Output is a classification result per supported category
        return SupportedCategories is not null && SupportedCategories.Length > 0
            ? new[] { SupportedCategories.Length }
            : Array.Empty<int>();
    }

    /// <inheritdoc/>
    public virtual DynamicShapeInfo GetDynamicShapeInfo()
    {
        return DynamicShapeInfo.None;
    }


    /// <inheritdoc/>
    public virtual void SaveModel(string filePath)
    {
        Helpers.ModelPersistenceGuard.EnforceBeforeSave();

        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        var fullPath = Path.GetFullPath(filePath);
        var directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        byte[] data = Serialize();
        byte[] envelopedData = ModelFileHeader.WrapWithHeader(
            data, this, GetInputShape(), GetOutputShape(), SerializationFormat.Json);
        File.WriteAllBytes(fullPath, envelopedData);
    }

    /// <inheritdoc/>
    public virtual void LoadModel(string filePath)
    {
        Helpers.ModelPersistenceGuard.EnforceBeforeLoad();

        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        byte[] data = File.ReadAllBytes(Path.GetFullPath(filePath));

        // Extract payload from AIMF envelope if present; use raw bytes for legacy files
        if (ModelFileHeader.HasHeader(data))
        {
            data = ModelFileHeader.ExtractPayload(data);
        }

        Deserialize(data);
    }

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
            vector = (Vector<T>)Engine.Divide(vector, sum);
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
