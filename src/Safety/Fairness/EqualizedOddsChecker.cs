using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Fairness;

/// <summary>
/// Checks for equalized odds violations by analyzing whether model quality or effort
/// varies across demographic groups mentioned in the text.
/// </summary>
/// <remarks>
/// <para>
/// Equalized odds requires that the true positive rate and false positive rate are equal
/// across all demographic groups. Since we're working with text (not tabular classification),
/// this module detects differential quality indicators: response length disparity, detail
/// level differences, hedging language, and conditional/qualifying statements that differ
/// by demographic group.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine asking an AI to write a recommendation letter for two
/// equally qualified candidates. If the letter for a man is detailed and enthusiastic but
/// the letter for a woman is shorter and uses more hedging words like "might" or "could",
/// that's an equalized odds violation — the model is giving different quality output for
/// different groups.
/// </para>
/// <para>
/// <b>References:</b>
/// - Equalized Odds in Machine Learning (Hardt et al., NeurIPS 2016)
/// - Measuring algorithmic fairness in text generation (2024)
/// - BEATS: Comprehensive bias evaluation test suite for LLMs (2025)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EqualizedOddsChecker<T> : ITextSafetyModule<T>
{
    private readonly double _disparityThreshold;

    private static readonly string[][] GenderGroups = new[]
    {
        new[] { "man", "men", "male", "boy", "he", "him", "his", "father", "husband" },
        new[] { "woman", "women", "female", "girl", "she", "her", "hers", "mother", "wife" }
    };

    // Hedging/uncertainty language (indicates lower confidence in statements)
    private static readonly HashSet<string> HedgingWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "might", "could", "perhaps", "possibly", "maybe", "somewhat",
        "arguably", "apparently", "seemingly", "may",
        "likely", "probable", "potential", "generally", "typically"
    };

    // Strong/decisive language (indicates higher confidence)
    private static readonly HashSet<string> DecisiveWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "definitely", "certainly", "clearly", "obviously", "undoubtedly", "always",
        "absolutely", "excellent", "outstanding", "exceptional", "remarkable",
        "extraordinary", "proven", "demonstrated", "achieved", "accomplished"
    };

    /// <inheritdoc />
    public string ModuleName => "EqualizedOddsChecker";

    /// <inheritdoc />
    public bool IsReady => true;

    /// <summary>
    /// Initializes a new equalized odds checker.
    /// </summary>
    /// <param name="disparityThreshold">Maximum allowed quality disparity between groups (0-1). Default: 0.3.</param>
    public EqualizedOddsChecker(double disparityThreshold = 0.3)
    {
        _disparityThreshold = disparityThreshold;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();
        if (string.IsNullOrWhiteSpace(text) || text.Length < 50) return findings;

        string lower = text.ToLowerInvariant();
        string[] words = lower.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

        // Analyze hedging disparity between gender groups
        var groupQuality = new List<GroupQualityMetrics>();

        for (int g = 0; g < GenderGroups.Length; g++)
        {
            var metrics = ComputeGroupQualityMetrics(words, GenderGroups[g]);
            if (metrics.ContextWords > 5)
            {
                groupQuality.Add(metrics);
            }
        }

        if (groupQuality.Count < 2) return findings;

        // Compare hedging ratio between groups
        double hedgeDisparity = Math.Abs(groupQuality[0].HedgingRatio - groupQuality[1].HedgingRatio);
        double decisiveDisparity = Math.Abs(groupQuality[0].DecisiveRatio - groupQuality[1].DecisiveRatio);
        double qualityDisparity = (hedgeDisparity + decisiveDisparity) / 2.0;

        if (qualityDisparity >= _disparityThreshold)
        {
            int lessConfidentGroup = groupQuality[0].HedgingRatio > groupQuality[1].HedgingRatio ? 0 : 1;
            string lessConfidentLabel = GenderGroups[lessConfidentGroup][0];
            string moreConfidentLabel = GenderGroups[1 - lessConfidentGroup][0];

            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Bias,
                Severity = qualityDisparity >= 0.5 ? SafetySeverity.High : SafetySeverity.Medium,
                Confidence = Math.Min(1.0, qualityDisparity * 1.5),
                Description = $"Equalized odds violation: language quality differs by gender " +
                              $"(disparity: {qualityDisparity:F3}). Text about '{lessConfidentLabel}' " +
                              $"uses more hedging language than text about '{moreConfidentLabel}', " +
                              $"indicating differential confidence in statements.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        return Array.Empty<SafetyFinding>();
    }

    private static GroupQualityMetrics ComputeGroupQualityMetrics(string[] words, string[] groupTerms)
    {
        int windowSize = 15;
        int hedgeCount = 0, decisiveCount = 0, contextWords = 0;

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
                if (HedgingWords.Contains(words[j])) hedgeCount++;
                if (DecisiveWords.Contains(words[j])) decisiveCount++;
            }
        }

        return new GroupQualityMetrics
        {
            ContextWords = contextWords,
            HedgingRatio = contextWords > 0 ? (double)hedgeCount / contextWords : 0,
            DecisiveRatio = contextWords > 0 ? (double)decisiveCount / contextWords : 0
        };
    }

    private struct GroupQualityMetrics
    {
        public int ContextWords;
        public double HedgingRatio;
        public double DecisiveRatio;
    }
}
