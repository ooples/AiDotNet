using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Fairness;

/// <summary>
/// Detects stereotypical associations between demographic groups and attributes in text.
/// </summary>
/// <remarks>
/// <para>
/// Identifies text that reinforces harmful stereotypes by detecting co-occurrence of demographic
/// group terms with stereotype-associated attributes. Uses a curated database of known stereotype
/// patterns across gender, racial, age, and other demographic dimensions.
/// </para>
/// <para>
/// <b>For Beginners:</b> Stereotypes are oversimplified beliefs about groups of people
/// (e.g., "women are emotional" or "elderly people are slow with technology"). This module
/// detects when AI output reinforces such stereotypes, even when the language is subtle.
/// </para>
/// <para>
/// <b>References:</b>
/// - StereoSet: Measuring stereotypical bias in pretrained language models (ACL 2021)
/// - CrowS-Pairs: Challenging dataset for measuring social biases (EMNLP 2020)
/// - BEATS: Comprehensive bias evaluation test suite for LLMs (2025)
/// - SB-Bench: Stereotype bias benchmark for multimodal models (2025)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class StereotypeDetector<T> : ITextSafetyModule<T>
{
    private readonly double _threshold;

    // Stereotype patterns: (demographic group terms, stereotypical attributes, anti-stereotypical attributes)
    private static readonly StereotypePattern[] Patterns = new[]
    {
        // Gender stereotypes
        new StereotypePattern(
            "gender",
            new[] { "woman", "women", "female", "girl", "she", "her" },
            new[] { "emotional", "nurturing", "submissive", "domestic", "gossip", "shopping",
                    "cooking", "cleaning", "beauty", "fashion", "secretary", "nurse", "irrational",
                    "hysterical", "nagging", "bossy", "catty", "ditzy" }),
        new StereotypePattern(
            "gender",
            new[] { "man", "men", "male", "boy", "he", "him" },
            new[] { "aggressive", "dominant", "stoic", "unemotional", "breadwinner", "provider",
                    "logical", "mechanical", "sports", "violent", "predatory", "womanizer" }),

        // Racial stereotypes
        new StereotypePattern(
            "race",
            new[] { "asian", "chinese", "japanese", "korean" },
            new[] { "math", "studious", "quiet", "submissive", "model minority", "tiger parent",
                    "martial arts", "nerdy", "hardworking", "overachiever" }),
        new StereotypePattern(
            "race",
            new[] { "black", "african american", "african" },
            new[] { "athletic", "aggressive", "criminal", "lazy", "loud", "dangerous",
                    "uneducated", "welfare", "ghetto", "thug", "gangster" }),
        new StereotypePattern(
            "race",
            new[] { "hispanic", "latino", "latina", "mexican" },
            new[] { "lazy", "illegal", "criminal", "passionate", "fiery", "spicy",
                    "uneducated", "poor", "gang", "immigrant" }),

        // Age stereotypes
        new StereotypePattern(
            "age",
            new[] { "old", "elderly", "senior", "aged", "retired" },
            new[] { "slow", "senile", "forgetful", "technophobe", "grumpy", "stubborn",
                    "out of touch", "feeble", "helpless", "burden", "confused" }),
        new StereotypePattern(
            "age",
            new[] { "young", "youth", "teenager", "millennial", "gen z" },
            new[] { "lazy", "entitled", "narcissistic", "addicted", "irresponsible",
                    "naive", "immature", "snowflake", "attention seeking" }),

        // Religion stereotypes
        new StereotypePattern(
            "religion",
            new[] { "muslim", "islamic", "arab" },
            new[] { "terrorist", "extremist", "oppressive", "backward", "violent", "radical",
                    "fundamentalist", "barbaric" }),
    };

    /// <inheritdoc />
    public string ModuleName => "StereotypeDetector";

    /// <inheritdoc />
    public bool IsReady => true;

    /// <summary>
    /// Initializes a new stereotype detector.
    /// </summary>
    /// <param name="threshold">Detection threshold (0-1). Default: 0.5.</param>
    public StereotypeDetector(double threshold = 0.5)
    {
        _threshold = threshold;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();
        if (string.IsNullOrWhiteSpace(text) || text.Length < 10) return findings;

        string lower = text.ToLowerInvariant();

        foreach (var pattern in Patterns)
        {
            double score = ComputeStereotypeScore(lower, pattern);
            if (score >= _threshold)
            {
                string groupLabel = pattern.GroupTerms.Length > 0 ? pattern.GroupTerms[0] : "unknown";
                findings.Add(new SafetyFinding
                {
                    Category = SafetyCategory.Bias,
                    Severity = score >= 0.8 ? SafetySeverity.High : SafetySeverity.Medium,
                    Confidence = Math.Min(1.0, score),
                    Description = $"Stereotypical association detected for '{pattern.AttributeCategory}' " +
                                  $"group '{groupLabel}' (score: {score:F3}). " +
                                  $"Text may reinforce harmful stereotypes about this group.",
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

    private static double ComputeStereotypeScore(string text, StereotypePattern pattern)
    {
        // Check if any group terms are present
        bool hasGroupTerm = false;
        foreach (string term in pattern.GroupTerms)
        {
            if (text.Contains(term))
            {
                hasGroupTerm = true;
                break;
            }
        }
        if (!hasGroupTerm) return 0;

        // Count stereotype attribute co-occurrences
        int stereotypeCount = 0;
        foreach (string attr in pattern.StereotypeAttributes)
        {
            if (text.Contains(attr)) stereotypeCount++;
        }

        if (stereotypeCount == 0) return 0;

        // Score based on number of stereotype attributes found near group terms
        // More attributes = higher confidence this is stereotyping
        double baseScore = Math.Min(1.0, stereotypeCount / 3.0);

        // Check proximity: are stereotype terms near group terms?
        double proximityBoost = ComputeProximityBoost(text, pattern.GroupTerms, pattern.StereotypeAttributes);

        return Math.Min(1.0, baseScore * (0.5 + 0.5 * proximityBoost));
    }

    private static double ComputeProximityBoost(string text, string[] groupTerms, string[] stereotypeTerms)
    {
        string[] words = text.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
        int windowSize = 15;
        int nearCount = 0;
        int totalStereotypes = 0;

        for (int i = 0; i < words.Length; i++)
        {
            bool isGroupTerm = false;
            foreach (string gt in groupTerms)
            {
                if (words[i].Contains(gt)) { isGroupTerm = true; break; }
            }
            if (!isGroupTerm) continue;

            // Check window around group term for stereotype terms
            int start = Math.Max(0, i - windowSize);
            int end = Math.Min(words.Length, i + windowSize + 1);
            for (int j = start; j < end; j++)
            {
                if (j == i) continue;
                foreach (string st in stereotypeTerms)
                {
                    if (words[j].Contains(st))
                    {
                        nearCount++;
                        break;
                    }
                }
            }
            totalStereotypes++;
        }

        return totalStereotypes > 0 ? Math.Min(1.0, (double)nearCount / totalStereotypes) : 0;
    }

    private sealed class StereotypePattern
    {
        public string AttributeCategory { get; }
        public string[] GroupTerms { get; }
        public string[] StereotypeAttributes { get; }

        public StereotypePattern(string category, string[] groupTerms, string[] stereotypeAttributes)
        {
            AttributeCategory = category;
            GroupTerms = groupTerms;
            StereotypeAttributes = stereotypeAttributes;
        }
    }
}
