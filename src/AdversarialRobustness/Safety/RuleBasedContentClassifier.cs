using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Serialization;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using AiDotNet.Validation;

namespace AiDotNet.AdversarialRobustness.Safety;

/// <summary>
/// A rule-based content classifier that uses pattern matching for classification.
/// </summary>
/// <remarks>
/// <para>
/// This classifier serves as a baseline or fallback when ML models are not available.
/// It uses configurable regex patterns to detect various categories of harmful content.
/// </para>
/// <para><b>For Beginners:</b> This is a simple classifier that looks for specific words
/// and patterns in text. While it's less sophisticated than ML-based classifiers, it's
/// fast, interpretable, and doesn't require training data.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class RuleBasedContentClassifier<T> : ContentClassifierBase<T>
{
    /// <summary>
    /// Timeout for regex operations to prevent ReDoS attacks.
    /// </summary>
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromMilliseconds(100);

    /// <summary>
    /// Pattern rules for each category.
    /// </summary>
    private Dictionary<string, List<string>> _categoryPatterns;

    /// <summary>
    /// Whether the classifier is initialized and ready.
    /// </summary>
    private bool _isReady;

    /// <summary>
    /// Initializes a new rule-based content classifier with default patterns.
    /// </summary>
    /// <param name="threshold">Detection threshold (default: 0.5).</param>
    public RuleBasedContentClassifier(double threshold = 0.5)
        : base(threshold)
    {
        _categoryPatterns = new Dictionary<string, List<string>>();
        InitializeDefaultPatterns();
        _isReady = true;
    }

    /// <summary>
    /// Initializes a new rule-based content classifier with custom patterns.
    /// </summary>
    /// <param name="categoryPatterns">Dictionary mapping categories to their detection patterns.</param>
    /// <param name="threshold">Detection threshold (default: 0.5).</param>
    public RuleBasedContentClassifier(
        Dictionary<string, List<string>> categoryPatterns,
        double threshold = 0.5)
        : base(threshold)
    {
        Guard.NotNull(categoryPatterns);
        _categoryPatterns = categoryPatterns;
        SupportedCategories = _categoryPatterns.Keys.ToArray();
        _isReady = true;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>Important:</b> Vector classification is limited for rule-based pattern matching.
    /// This classifier uses regex patterns that match text content. When classifying vectors,
    /// the original text content is not recoverable, so pattern matching cannot detect
    /// harmful content effectively.
    /// </para>
    /// <para>
    /// For reliable harmful content detection, use <see cref="ClassifyText(string)"/> instead.
    /// </para>
    /// </remarks>
    public override ContentClassificationResult<T> Classify(Vector<T> content)
    {
        if (content == null)
        {
            throw new ArgumentNullException(nameof(content));
        }

        // Rule-based pattern matching requires original text, not numeric vector representations.
        // Vector classification cannot effectively detect harmful content since regex patterns
        // are designed to match text strings (e.g., "kill", "hate"), not numeric values.
        // Return a "safe" result with zero confidence to indicate no meaningful classification was possible.
        return new ContentClassificationResult<T>
        {
            IsHarmful = false,
            OverallConfidence = NumOps.Zero, // Zero confidence indicates no detection was possible
            PrimaryCategory = "Unknown",
            DetectedCategories = Array.Empty<string>(),
            RecommendedAction = "Review",
            CategoryScores = new Dictionary<string, T>()
        };
    }

    /// <inheritdoc/>
    public override ContentClassificationResult<T> ClassifyText(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return new ContentClassificationResult<T>
            {
                IsHarmful = false,
                OverallConfidence = NumOps.One,
                PrimaryCategory = "Safe",
                RecommendedAction = "Allow"
            };
        }

        return ClassifyTextInternal(text);
    }

    /// <inheritdoc/>
    public override bool IsReady() => _isReady;

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        var data = new SerializationData
        {
            Threshold = NumOps.ToDouble(DetectionThreshold),
            CategoryPatterns = _categoryPatterns
        };

        var json = JsonConvert.SerializeObject(data, Formatting.None);
        return Encoding.UTF8.GetBytes(json);
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] data)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        var json = Encoding.UTF8.GetString(data);

        var settings = new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.Auto,
            SerializationBinder = new SafeSerializationBinder()
        };

        var deserialized = JsonConvert.DeserializeObject<SerializationData>(json, settings);
        if (deserialized != null)
        {
            DetectionThreshold = NumOps.FromDouble(deserialized.Threshold);
            _categoryPatterns = deserialized.CategoryPatterns ?? new Dictionary<string, List<string>>();
            SupportedCategories = _categoryPatterns.Keys.ToArray();
        }

        _isReady = true;
    }

    /// <inheritdoc/>
    public override void SaveModel(string filePath)
    {
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

        File.WriteAllBytes(fullPath, Serialize());
    }

    /// <inheritdoc/>
    public override void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        var fullPath = Path.GetFullPath(filePath);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException("Model file not found.", fullPath);
        }

        Deserialize(File.ReadAllBytes(fullPath));
    }

    /// <summary>
    /// Adds a detection pattern for a category.
    /// </summary>
    /// <param name="category">The category name.</param>
    /// <param name="pattern">The regex pattern to add.</param>
    public void AddPattern(string category, string pattern)
    {
        if (string.IsNullOrWhiteSpace(category))
        {
            throw new ArgumentException("Category cannot be null or empty.", nameof(category));
        }

        if (string.IsNullOrWhiteSpace(pattern))
        {
            throw new ArgumentException("Pattern cannot be null or empty.", nameof(pattern));
        }

        if (!_categoryPatterns.ContainsKey(category))
        {
            _categoryPatterns[category] = new List<string>();
            SupportedCategories = _categoryPatterns.Keys.ToArray();
        }

        _categoryPatterns[category].Add(pattern);
    }

    /// <summary>
    /// Removes all patterns for a category.
    /// </summary>
    /// <param name="category">The category to clear.</param>
    public void ClearCategory(string category)
    {
        if (_categoryPatterns.TryGetValue(category, out var patterns))
        {
            patterns.Clear();
        }
    }

    private ContentClassificationResult<T> ClassifyTextInternal(string text)
    {
        text = text.ToLowerInvariant();
        var categoryScores = new Dictionary<string, T>();

        foreach (var category in _categoryPatterns.Keys)
        {
            var patterns = _categoryPatterns[category];
            int matchCount = 0;
            int totalPatterns = patterns.Count;

            if (totalPatterns == 0)
            {
                categoryScores[category] = NumOps.Zero;
                continue;
            }

            foreach (var pattern in patterns)
            {
                try
                {
                    if (Regex.IsMatch(text, pattern, RegexOptions.IgnoreCase, RegexTimeout))
                    {
                        matchCount++;
                    }
                }
                catch (RegexMatchTimeoutException)
                {
                    // Pattern timed out, skip
                }
            }

            // Score is proportion of patterns matched
            double score = (double)matchCount / totalPatterns;
            categoryScores[category] = NumOps.FromDouble(score);
        }

        return CreateResultFromScores(categoryScores);
    }

    private void InitializeDefaultPatterns()
    {
        _categoryPatterns = new Dictionary<string, List<string>>
        {
            ["Toxic"] = new List<string>
            {
                @"\b(stupid|idiot|moron|dumb)\b",
                @"\b(hate|despise|loathe)\b",
                @"\b(shut up|stfu)\b"
            },
            ["Violence"] = new List<string>
            {
                @"\b(kill|murder|assassinate)\b",
                @"\b(weapon|gun|bomb)\b",
                @"\b(attack|assault|beat)\b",
                @"\b(destroy|demolish)\b"
            },
            ["HateSpeech"] = new List<string>
            {
                @"\b(racist|sexist)\b",
                @"\b(discriminat)\b",
                @"\b(supremac)\b"
            },
            ["AdultContent"] = new List<string>
            {
                @"\b(nsfw|explicit)\b",
                @"\b(adult only)\b"
            },
            ["Harassment"] = new List<string>
            {
                @"\b(threaten)\b",
                @"\b(stalk)\b",
                @"\b(bully)\b"
            },
            ["SelfHarm"] = new List<string>
            {
                @"\b(suicide|suicidal)\b",
                @"\b(self-harm|cut myself)\b"
            },
            ["PrivateInformation"] = new List<string>
            {
                @"\b\d{3}-\d{2}-\d{4}\b", // SSN pattern
                @"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", // Credit card
                @"\bpassword\s*[:=]\s*\S+\b"
            }
        };

        SupportedCategories = _categoryPatterns.Keys.ToArray();
    }

    /// <summary>
    /// Serialization data structure.
    /// </summary>
    private class SerializationData
    {
        public double Threshold { get; set; }
        public Dictionary<string, List<string>>? CategoryPatterns { get; set; }
    }
}
