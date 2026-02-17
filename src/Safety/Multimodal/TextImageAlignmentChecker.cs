using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Multimodal;

/// <summary>
/// Checks semantic alignment between text descriptions and associated images to detect
/// mismatched or deceptive text-image pairs.
/// </summary>
/// <remarks>
/// <para>
/// Analyzes whether the visual content of an image is consistent with its text description.
/// Uses feature extraction from both modalities to compute alignment scores. Misalignment
/// can indicate deceptive content (safe text paired with unsafe image), phishing, or
/// misinformation.
/// </para>
/// <para>
/// <b>For Beginners:</b> This module checks if what an image shows matches what the text
/// says about it. For example, if someone labels an image "cute puppy" but the image shows
/// something violent, this module catches that mismatch.
/// </para>
/// <para>
/// <b>References:</b>
/// - CLIP: Learning transferable visual models from natural language (OpenAI, 2021)
/// - OmniSafeBench-MM: Multimodal safety evaluation (2025)
/// - MM-SafetyBench: 13 scenarios for multimodal safety (ECCV 2024)
/// - Cross-modal jailbreak attacks on multimodal LLMs (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TextImageAlignmentChecker<T> : ITextSafetyModule<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _mismatchThreshold;

    // Keywords for visual content categories
    private static readonly Dictionary<string, string[]> VisualCategoryKeywords = new()
    {
        ["safe"] = new[] { "cute", "puppy", "kitten", "flower", "landscape", "sunset", "family",
            "baby", "garden", "nature", "beautiful", "peaceful", "happy", "smile" },
        ["violent"] = new[] { "blood", "weapon", "gun", "knife", "fight", "war", "explosion",
            "attack", "injury", "wound", "dead", "kill", "murder", "combat" },
        ["nsfw"] = new[] { "nude", "naked", "explicit", "sexual", "pornograph", "erotic",
            "adult content", "xxx", "nsfw" },
        ["hateful"] = new[] { "hate", "racist", "supremac", "nazi", "genocide", "ethnic cleansing",
            "slur", "bigot", "discriminat" }
    };

    /// <inheritdoc />
    public string ModuleName => "TextImageAlignmentChecker";

    /// <inheritdoc />
    public bool IsReady => true;

    /// <summary>
    /// Initializes a new text-image alignment checker.
    /// </summary>
    /// <param name="mismatchThreshold">Threshold for flagging misalignment (0-1). Default: 0.5.</param>
    public TextImageAlignmentChecker(double mismatchThreshold = 0.5)
    {
        _mismatchThreshold = mismatchThreshold;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();
        if (string.IsNullOrWhiteSpace(text)) return findings;

        string lower = text.ToLowerInvariant();

        // Detect text that explicitly claims safety while containing unsafe terms
        bool claimsSafe = ContainsAny(lower, new[] { "safe", "harmless", "innocent", "clean", "appropriate" });
        bool hasUnsafeContent = ContainsAny(lower, VisualCategoryKeywords["violent"]) ||
                                ContainsAny(lower, VisualCategoryKeywords["nsfw"]) ||
                                ContainsAny(lower, VisualCategoryKeywords["hateful"]);

        if (claimsSafe && hasUnsafeContent)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Manipulated,
                Severity = SafetySeverity.Medium,
                Confidence = 0.7,
                Description = "Text-image alignment concern: text claims content is safe but contains " +
                              "unsafe content descriptors. This may indicate an attempt to disguise " +
                              "harmful content with a safe-sounding description.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <summary>
    /// Evaluates alignment between text and an associated image.
    /// </summary>
    /// <param name="text">Text description or caption.</param>
    /// <param name="image">Associated image tensor.</param>
    /// <returns>Alignment findings.</returns>
    public IReadOnlyList<SafetyFinding> EvaluateAlignment(string text, Tensor<T> image)
    {
        var findings = new List<SafetyFinding>();
        if (string.IsNullOrWhiteSpace(text)) return findings;

        var span = image.Data.Span;
        if (span.Length < 16) return findings;

        string lower = text.ToLowerInvariant();

        // Extract text category signals
        var textCategories = ClassifyTextContent(lower);

        // Extract image visual signals
        var imageFeatures = ExtractImageFeatures(span, image.Shape);

        // Check for misalignment between text claims and image content
        // Text says "safe" but image has unsafe visual features
        if (textCategories.ContainsKey("safe") && textCategories["safe"] > 0.5)
        {
            if (imageFeatures.SkinFraction > 0.4 || imageFeatures.RedDominance > 0.3)
            {
                double mismatchScore = Math.Max(imageFeatures.SkinFraction, imageFeatures.RedDominance);
                if (mismatchScore >= _mismatchThreshold)
                {
                    findings.Add(new SafetyFinding
                    {
                        Category = SafetyCategory.Manipulated,
                        Severity = SafetySeverity.High,
                        Confidence = Math.Min(1.0, mismatchScore),
                        Description = $"Text-image misalignment (score: {mismatchScore:F3}). " +
                                      $"Text describes safe content but image features suggest otherwise " +
                                      $"(skin fraction: {imageFeatures.SkinFraction:F3}, " +
                                      $"red dominance: {imageFeatures.RedDominance:F3}).",
                        RecommendedAction = SafetyAction.Warn,
                        SourceModule = ModuleName
                    });
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

    private static Dictionary<string, double> ClassifyTextContent(string text)
    {
        var scores = new Dictionary<string, double>();

        foreach (var kvp in VisualCategoryKeywords)
        {
            int matches = 0;
            foreach (string keyword in kvp.Value)
            {
                if (text.Contains(keyword)) matches++;
            }
            scores[kvp.Key] = Math.Min(1.0, (double)matches / 3);
        }

        return scores;
    }

    private static ImageVisualFeatures ExtractImageFeatures(ReadOnlySpan<T> span, int[] shape)
    {
        int totalPixels = span.Length;
        int channels = shape.Length >= 3 ? shape[0] : 1;
        int pixelsPerChannel = channels > 0 ? totalPixels / channels : totalPixels;

        double skinCount = 0, redCount = 0, darkCount = 0;
        int analyzed = 0;

        // Sample pixels
        int step = Math.Max(1, pixelsPerChannel / 1000);
        for (int i = 0; i < pixelsPerChannel; i += step)
        {
            double r, g, b;
            if (channels >= 3)
            {
                r = NumOps.ToDouble(span[i]);
                g = i + pixelsPerChannel < totalPixels ? NumOps.ToDouble(span[i + pixelsPerChannel]) : r;
                b = i + 2 * pixelsPerChannel < totalPixels ? NumOps.ToDouble(span[i + 2 * pixelsPerChannel]) : r;
            }
            else
            {
                r = g = b = NumOps.ToDouble(span[i]);
            }

            if (r <= 1.0 && g <= 1.0 && b <= 1.0) { r *= 255; g *= 255; b *= 255; }

            // Skin detection (simplified)
            if (r > 95 && g > 40 && b > 20 && r > g && r > b && Math.Abs(r - g) > 15)
                skinCount++;

            // Red dominance
            if (r > 150 && r > g * 1.5 && r > b * 1.5)
                redCount++;

            // Dark regions
            if (r < 50 && g < 50 && b < 50)
                darkCount++;

            analyzed++;
        }

        if (analyzed == 0) return new ImageVisualFeatures();

        return new ImageVisualFeatures
        {
            SkinFraction = skinCount / analyzed,
            RedDominance = redCount / analyzed,
            DarkFraction = darkCount / analyzed
        };
    }

    private static bool ContainsAny(string text, string[] terms)
    {
        foreach (string term in terms)
        {
            if (text.Contains(term)) return true;
        }
        return false;
    }

    private struct ImageVisualFeatures
    {
        public double SkinFraction;
        public double RedDominance;
        public double DarkFraction;
    }
}
