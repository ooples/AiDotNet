using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Fairness;

/// <summary>
/// Detects intersectional bias — bias that uniquely affects individuals at the intersection
/// of multiple demographic identities (e.g., Black women, elderly Asian men).
/// </summary>
/// <remarks>
/// <para>
/// Intersectional bias occurs when the combined effect of belonging to multiple demographic
/// groups produces worse outcomes than would be predicted by looking at each group individually.
/// For example, a model might generate positive text about "women" and about "Black people"
/// separately, but generate negative text about "Black women" specifically. This module detects
/// such compounding bias by analyzing sentiment around intersectional identity mentions.
/// </para>
/// <para>
/// <b>For Beginners:</b> A person is not just their gender or their race — they are both at
/// the same time. A Black woman may face unique biases that differ from what Black men or
/// White women experience. This detector looks for cases where AI output is specifically
/// biased against people with multiple overlapping identities.
/// </para>
/// <para>
/// <b>References:</b>
/// - Demarginalizing the intersection of race and sex (Crenshaw, 1989)
/// - Intersectional bias in AI fairness (Buolamwini &amp; Gebru, Gender Shades, 2018)
/// - BEATS: Comprehensive bias evaluation test suite for LLMs (2025, arxiv:2503.24310)
/// - FLEX: Robustness of fairness evaluation under adversarial prompts (NAACL 2025)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class IntersectionalBiasDetector<T> : ITextSafetyModule<T>
{
    private readonly double _threshold;

    // Identity axes and their group terms
    private static readonly Dictionary<string, string[][]> IdentityAxes = new(StringComparer.OrdinalIgnoreCase)
    {
        ["gender"] = new[]
        {
            new[] { "man", "men", "male", "boy", "he", "him", "his" },
            new[] { "woman", "women", "female", "girl", "she", "her", "hers" }
        },
        ["race"] = new[]
        {
            new[] { "white", "caucasian" },
            new[] { "black", "african", "african american" },
            new[] { "asian", "east asian", "south asian" },
            new[] { "hispanic", "latino", "latina" }
        },
        ["age"] = new[]
        {
            new[] { "young", "youth", "teenager" },
            new[] { "old", "elderly", "senior", "aged" }
        }
    };

    // Positive and negative sentiment words for scoring
    private static readonly HashSet<string> PositiveWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "good", "great", "excellent", "positive", "successful", "intelligent", "smart",
        "capable", "skilled", "talented", "hardworking", "reliable", "trustworthy",
        "competent", "professional", "leader", "strong", "creative", "innovative",
        "brilliant", "accomplished", "respected", "admired", "beautiful", "kind"
    };

    private static readonly HashSet<string> NegativeWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "bad", "poor", "terrible", "negative", "lazy", "stupid", "incompetent",
        "unreliable", "criminal", "dangerous", "aggressive", "weak", "ignorant",
        "inferior", "subordinate", "emotional", "irrational", "primitive", "threatening",
        "suspicious", "untrustworthy", "dishonest", "ugly", "cruel", "angry"
    };

    /// <inheritdoc />
    public string ModuleName => "IntersectionalBiasDetector";

    /// <inheritdoc />
    public bool IsReady => true;

    /// <summary>
    /// Initializes a new intersectional bias detector.
    /// </summary>
    /// <param name="threshold">Detection threshold for bias findings (0.0-1.0). Default: 0.3.</param>
    public IntersectionalBiasDetector(double threshold = 0.3)
    {
        if (threshold < 0 || threshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold),
                "Threshold must be between 0 and 1.");
        }

        _threshold = threshold;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();
        if (string.IsNullOrWhiteSpace(text) || text.Length < 30) return findings;

        string lower = text.ToLowerInvariant();
        string[] words = lower.Split(new[] { ' ', '\t', '\n', '\r', '.', ',', ';', ':', '!', '?', '(', ')' },
            StringSplitOptions.RemoveEmptyEntries);

        // Compute single-axis sentiment for each group
        var singleAxisSentiments = new Dictionary<string, double>();
        foreach (var axis in IdentityAxes)
        {
            for (int g = 0; g < axis.Value.Length; g++)
            {
                double sentiment = ComputeGroupSentiment(words, axis.Value[g]);
                if (!double.IsNaN(sentiment))
                {
                    string key = $"{axis.Key}:{axis.Value[g][0]}";
                    singleAxisSentiments[key] = sentiment;
                }
            }
        }

        // Now compute intersectional sentiments (pairs of axes)
        string[] axisNames = new string[IdentityAxes.Count];
        IdentityAxes.Keys.CopyTo(axisNames, 0);

        for (int a1 = 0; a1 < axisNames.Length; a1++)
        {
            for (int a2 = a1 + 1; a2 < axisNames.Length; a2++)
            {
                var groups1 = IdentityAxes[axisNames[a1]];
                var groups2 = IdentityAxes[axisNames[a2]];

                for (int g1 = 0; g1 < groups1.Length; g1++)
                {
                    for (int g2 = 0; g2 < groups2.Length; g2++)
                    {
                        // Find windows where both group terms co-occur
                        double intersectionalSentiment = ComputeIntersectionalSentiment(
                            words, groups1[g1], groups2[g2]);

                        if (double.IsNaN(intersectionalSentiment)) continue;

                        // Get single-axis baselines
                        string key1 = $"{axisNames[a1]}:{groups1[g1][0]}";
                        string key2 = $"{axisNames[a2]}:{groups2[g2][0]}";

                        double baseline1 = singleAxisSentiments.TryGetValue(key1, out var s1) ? s1 : 0;
                        double baseline2 = singleAxisSentiments.TryGetValue(key2, out var s2) ? s2 : 0;

                        // Expected intersectional sentiment: average of single-axis sentiments
                        double expectedSentiment = (baseline1 + baseline2) / 2.0;

                        // Intersectional bias: actual sentiment is worse than expected
                        double biasScore = expectedSentiment - intersectionalSentiment;

                        if (biasScore >= _threshold)
                        {
                            string label1 = groups1[g1][0];
                            string label2 = groups2[g2][0];

                            findings.Add(new SafetyFinding
                            {
                                Category = SafetyCategory.Bias,
                                Severity = biasScore >= 0.6 ? SafetySeverity.High :
                                           biasScore >= 0.4 ? SafetySeverity.Medium : SafetySeverity.Low,
                                Confidence = Math.Min(1.0, biasScore),
                                Description = $"Intersectional bias detected: '{label1} {label2}' " +
                                              $"(score: {biasScore:F3}). Text about this intersectional " +
                                              $"identity is more negative than expected from individual " +
                                              $"'{axisNames[a1]}' and '{axisNames[a2]}' sentiments " +
                                              $"(expected: {expectedSentiment:F3}, actual: {intersectionalSentiment:F3}).",
                                RecommendedAction = SafetyAction.Warn,
                                SourceModule = ModuleName
                            });
                        }
                    }
                }
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
        int windowSize = 12;
        int posCount = 0, negCount = 0, contextWords = 0;

        for (int i = 0; i < words.Length; i++)
        {
            bool isGroupTerm = false;
            foreach (string term in groupTerms)
            {
                if (words[i] == term || words[i].Contains(term))
                {
                    isGroupTerm = true;
                    break;
                }
            }
            if (!isGroupTerm) continue;

            int start = Math.Max(0, i - windowSize);
            int end = Math.Min(words.Length, i + windowSize + 1);

            for (int j = start; j < end; j++)
            {
                if (j == i) continue;
                contextWords++;
                if (PositiveWords.Contains(words[j])) posCount++;
                if (NegativeWords.Contains(words[j])) negCount++;
            }
        }

        if (contextWords == 0) return double.NaN;
        int total = posCount + negCount;
        if (total == 0) return 0;
        return (double)(posCount - negCount) / total;
    }

    private static double ComputeIntersectionalSentiment(string[] words, string[] group1Terms, string[] group2Terms)
    {
        int windowSize = 20;
        int posCount = 0, negCount = 0, contextWords = 0;

        // Find locations where both groups are mentioned within the same window
        for (int i = 0; i < words.Length; i++)
        {
            bool isGroup1 = false;
            foreach (string term in group1Terms)
            {
                if (words[i] == term || words[i].Contains(term))
                {
                    isGroup1 = true;
                    break;
                }
            }
            if (!isGroup1) continue;

            // Check if group2 is mentioned nearby
            int start = Math.Max(0, i - windowSize);
            int end = Math.Min(words.Length, i + windowSize + 1);

            bool group2Nearby = false;
            for (int j = start; j < end; j++)
            {
                if (j == i) continue;
                foreach (string term in group2Terms)
                {
                    if (words[j] == term || words[j].Contains(term))
                    {
                        group2Nearby = true;
                        break;
                    }
                }
                if (group2Nearby) break;
            }

            if (!group2Nearby) continue;

            // Both groups co-occur — analyze sentiment in this window
            for (int j = start; j < end; j++)
            {
                if (j == i) continue;
                contextWords++;
                if (PositiveWords.Contains(words[j])) posCount++;
                if (NegativeWords.Contains(words[j])) negCount++;
            }
        }

        if (contextWords == 0) return double.NaN;
        int total = posCount + negCount;
        if (total == 0) return 0;
        return (double)(posCount - negCount) / total;
    }
}
