using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Fairness;

/// <summary>
/// Checks for demographic parity violations by detecting differential treatment of
/// demographic groups in model outputs.
/// </summary>
/// <remarks>
/// <para>
/// Demographic parity requires that the probability of a positive outcome is the same
/// across all demographic groups. This module analyzes text for references to protected
/// attributes and detects sentiment/polarity differences that indicate bias.
/// </para>
/// <para>
/// <b>For Beginners:</b> This module checks whether AI output treats different groups of
/// people fairly. For example, if a model describes men positively but women negatively
/// when answering similar questions, that's a demographic parity violation.
/// </para>
/// <para>
/// <b>References:</b>
/// - BEATS: Comprehensive bias evaluation test suite for LLMs (2025)
/// - SB-Bench: Stereotype bias benchmark (2025)
/// - Demographic-targeted bias: race/ethnicity 55.6% exploitability (2025)
/// - Fairness metrics in machine learning survey (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DemographicParityChecker<T> : ITextSafetyModule<T>
{
    private readonly double _disparityThreshold;
    private readonly string[] _protectedAttributes;

    // Demographic group indicators keyed by attribute category
    private static readonly Dictionary<string, string[][]> DemographicGroups = new()
    {
        ["gender"] = new[]
        {
            new[] { "man", "men", "male", "boy", "father", "husband", "he", "him", "his" },
            new[] { "woman", "women", "female", "girl", "mother", "wife", "she", "her", "hers" }
        },
        ["race"] = new[]
        {
            new[] { "white", "caucasian", "european" },
            new[] { "black", "african", "african american" },
            new[] { "asian", "east asian", "south asian" },
            new[] { "hispanic", "latino", "latina", "latinx" }
        },
        ["age"] = new[]
        {
            new[] { "young", "youth", "teenager", "millennial", "gen z" },
            new[] { "old", "elderly", "senior", "aged", "retired", "boomer" }
        },
        ["religion"] = new[]
        {
            new[] { "christian", "catholic", "protestant" },
            new[] { "muslim", "islamic", "islam" },
            new[] { "jewish", "judaism", "hindu", "buddhist" }
        }
    };

    // Positive sentiment indicators
    private static readonly HashSet<string> PositiveWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "good", "great", "excellent", "positive", "successful", "intelligent", "smart",
        "capable", "skilled", "talented", "hardworking", "reliable", "trustworthy",
        "competent", "professional", "leader", "strong", "creative", "innovative"
    };

    // Negative sentiment indicators
    private static readonly HashSet<string> NegativeWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "bad", "poor", "terrible", "negative", "lazy", "stupid", "incompetent",
        "unreliable", "criminal", "dangerous", "aggressive", "weak", "ignorant",
        "inferior", "subordinate", "emotional", "irrational", "primitive"
    };

    /// <inheritdoc />
    public string ModuleName => "DemographicParityChecker";

    /// <inheritdoc />
    public bool IsReady => true;

    /// <summary>
    /// Initializes a new demographic parity checker.
    /// </summary>
    /// <param name="disparityThreshold">Maximum allowed sentiment disparity between groups (0-1). Default: 0.3.</param>
    /// <param name="protectedAttributes">Attribute categories to check. Default: ["gender", "race"].</param>
    public DemographicParityChecker(
        double disparityThreshold = 0.3,
        string[]? protectedAttributes = null)
    {
        _disparityThreshold = disparityThreshold;
        _protectedAttributes = protectedAttributes ?? new[] { "gender", "race" };
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();
        if (string.IsNullOrWhiteSpace(text) || text.Length < 20) return findings;

        string lower = text.ToLowerInvariant();
        string[] words = lower.Split(new[] { ' ', '\t', '\n', '\r', '.', ',', ';', ':', '!', '?', '(', ')' },
            StringSplitOptions.RemoveEmptyEntries);

        foreach (string attribute in _protectedAttributes)
        {
            if (!DemographicGroups.TryGetValue(attribute, out var groups)) continue;

            var groupSentiments = new List<double>();
            var groupNames = new List<string>();

            for (int g = 0; g < groups.Length; g++)
            {
                var groupTerms = groups[g];
                double sentiment = ComputeGroupSentiment(words, groupTerms);
                if (!double.IsNaN(sentiment))
                {
                    groupSentiments.Add(sentiment);
                    groupNames.Add(groupTerms[0]);
                }
            }

            if (groupSentiments.Count < 2) continue;

            // Find max disparity between any two groups
            double maxDisparity = 0;
            string highGroup = "", lowGroup = "";
            for (int i = 0; i < groupSentiments.Count; i++)
            {
                for (int j = i + 1; j < groupSentiments.Count; j++)
                {
                    double disparity = Math.Abs(groupSentiments[i] - groupSentiments[j]);
                    if (disparity > maxDisparity)
                    {
                        maxDisparity = disparity;
                        highGroup = groupSentiments[i] > groupSentiments[j] ? groupNames[i] : groupNames[j];
                        lowGroup = groupSentiments[i] > groupSentiments[j] ? groupNames[j] : groupNames[i];
                    }
                }
            }

            if (maxDisparity >= _disparityThreshold)
            {
                findings.Add(new SafetyFinding
                {
                    Category = SafetyCategory.Bias,
                    Severity = maxDisparity >= 0.6 ? SafetySeverity.High : SafetySeverity.Medium,
                    Confidence = Math.Min(1.0, maxDisparity),
                    Description = $"Demographic parity violation for '{attribute}' (disparity: {maxDisparity:F3}). " +
                                  $"Group '{highGroup}' is described more positively than '{lowGroup}'.",
                    RecommendedAction = SafetyAction.Warn,
                    SourceModule = ModuleName
                });
            }
        }

        return findings;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        return Array.Empty<SafetyFinding>();
    }

    private static double ComputeGroupSentiment(string[] words, string[] groupTerms)
    {
        // Find windows around group term mentions and compute sentiment
        int windowSize = 10;
        int posCount = 0, negCount = 0, totalContext = 0;

        for (int i = 0; i < words.Length; i++)
        {
            bool isGroupMention = false;
            foreach (string term in groupTerms)
            {
                if (words[i] == term || words[i].Contains(term))
                {
                    isGroupMention = true;
                    break;
                }
            }

            if (!isGroupMention) continue;

            // Analyze surrounding window
            int start = Math.Max(0, i - windowSize);
            int end = Math.Min(words.Length, i + windowSize + 1);

            for (int j = start; j < end; j++)
            {
                if (j == i) continue;
                if (PositiveWords.Contains(words[j])) posCount++;
                if (NegativeWords.Contains(words[j])) negCount++;
                totalContext++;
            }
        }

        if (totalContext == 0) return double.NaN;

        // Sentiment: -1 (all negative) to +1 (all positive)
        int total = posCount + negCount;
        if (total == 0) return 0;
        return (double)(posCount - negCount) / total;
    }
}
