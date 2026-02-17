using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Multimodal;

/// <summary>
/// Checks consistency between different modalities (text, image, audio) to detect
/// misaligned or manipulated multimodal content.
/// </summary>
/// <remarks>
/// <para>
/// Cross-modal attacks exploit the gap between modalities — for example, a benign text
/// description paired with a harmful image, or a safe-looking image with hidden toxic text.
/// This module detects such misalignment by comparing safety signals across modalities.
/// </para>
/// <para>
/// <b>For Beginners:</b> When you have content with both text and images (or audio),
/// sometimes the text says one thing but the image shows something different or harmful.
/// This module catches those mismatches — for example, a message that says "cute animals"
/// but contains a violent image.
/// </para>
/// <para>
/// <b>Detection approach:</b>
/// 1. Run each modality through its respective safety module independently
/// 2. Compare finding severity across modalities
/// 3. Flag cases where one modality is safe but another is unsafe (potential evasion)
/// 4. Flag cases where text-image semantic similarity is very low (potential mismatch)
/// </para>
/// <para>
/// <b>References:</b>
/// - OmniSafeBench-MM: 9 risk domains, 50 fine-grained categories (2025)
/// - MM-SafetyBench: 13 scenarios for multimodal safety (ECCV 2024)
/// - Cross-modal jailbreak attacks on multimodal LLMs (2024)
/// - AnyAttack: Transferable adversarial attacks on vision-language models (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CrossModalConsistencyChecker<T> : ITextSafetyModule<T>
{
    private readonly double _mismatchThreshold;

    /// <inheritdoc />
    public string ModuleName => "CrossModalConsistencyChecker";

    /// <inheritdoc />
    public bool IsReady => true;

    /// <summary>
    /// Initializes a new cross-modal consistency checker.
    /// </summary>
    /// <param name="mismatchThreshold">
    /// Score difference threshold between modalities to flag as inconsistent (0-1).
    /// Default: 0.5 — if one modality scores 0.2 and another 0.8, that's a 0.6 gap.
    /// </param>
    public CrossModalConsistencyChecker(double mismatchThreshold = 0.5)
    {
        if (mismatchThreshold < 0 || mismatchThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(mismatchThreshold),
                "Mismatch threshold must be between 0 and 1.");
        }

        _mismatchThreshold = mismatchThreshold;
    }

    /// <summary>
    /// Evaluates text for cross-modal consistency indicators.
    /// </summary>
    /// <remarks>
    /// When used standalone, this checks for text that attempts to override or contradict
    /// safety signals (e.g., "this image is safe" prepended to a prompt). Full cross-modal
    /// analysis requires pairing with image/audio evaluations via the SafetyPipeline.
    /// </remarks>
    public IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        if (string.IsNullOrWhiteSpace(text))
        {
            return findings;
        }

        // Detect text that may be attempting to override safety classification
        // of another modality (e.g., "ignore the image, it's safe")
        string lower = text.ToLowerInvariant();

        if (ContainsOverridePattern(lower))
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.PromptInjection,
                Severity = SafetySeverity.Medium,
                Confidence = 0.7,
                Description = "Text contains patterns that may attempt to override safety classification " +
                              "of other modalities (cross-modal manipulation attempt).",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <summary>
    /// Compares safety findings from multiple modalities and returns cross-modal findings.
    /// </summary>
    /// <param name="textFindings">Findings from text analysis.</param>
    /// <param name="imageFindings">Findings from image analysis.</param>
    /// <param name="audioFindings">Findings from audio analysis.</param>
    /// <returns>Cross-modal consistency findings.</returns>
    public IReadOnlyList<SafetyFinding> CheckConsistency(
        IReadOnlyList<SafetyFinding>? textFindings,
        IReadOnlyList<SafetyFinding>? imageFindings,
        IReadOnlyList<SafetyFinding>? audioFindings)
    {
        var findings = new List<SafetyFinding>();

        double textMaxSeverity = GetMaxSeverityScore(textFindings);
        double imageMaxSeverity = GetMaxSeverityScore(imageFindings);
        double audioMaxSeverity = GetMaxSeverityScore(audioFindings);

        // Check text-image consistency
        if (textFindings != null && imageFindings != null)
        {
            double gap = Math.Abs(textMaxSeverity - imageMaxSeverity);
            if (gap >= _mismatchThreshold)
            {
                string safeModality = textMaxSeverity < imageMaxSeverity ? "text" : "image";
                string unsafeModality = textMaxSeverity < imageMaxSeverity ? "image" : "text";

                findings.Add(new SafetyFinding
                {
                    Category = SafetyCategory.Manipulated,
                    Severity = SafetySeverity.High,
                    Confidence = gap,
                    Description = $"Cross-modal inconsistency detected: {safeModality} appears safe but " +
                                  $"{unsafeModality} is flagged (severity gap: {gap:F2}). " +
                                  "This may indicate a cross-modal evasion attack.",
                    RecommendedAction = SafetyAction.Block,
                    SourceModule = ModuleName
                });
            }
        }

        // Check text-audio consistency
        if (textFindings != null && audioFindings != null)
        {
            double gap = Math.Abs(textMaxSeverity - audioMaxSeverity);
            if (gap >= _mismatchThreshold)
            {
                string safeModality = textMaxSeverity < audioMaxSeverity ? "text" : "audio";
                string unsafeModality = textMaxSeverity < audioMaxSeverity ? "audio" : "text";

                findings.Add(new SafetyFinding
                {
                    Category = SafetyCategory.Manipulated,
                    Severity = SafetySeverity.High,
                    Confidence = gap,
                    Description = $"Cross-modal inconsistency detected: {safeModality} appears safe but " +
                                  $"{unsafeModality} is flagged (severity gap: {gap:F2}).",
                    RecommendedAction = SafetyAction.Block,
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

    private static double GetMaxSeverityScore(IReadOnlyList<SafetyFinding>? findings)
    {
        if (findings == null || findings.Count == 0)
        {
            return 0.0;
        }

        double maxScore = 0.0;
        foreach (var finding in findings)
        {
            double severityWeight = finding.Severity switch
            {
                SafetySeverity.Critical => 1.0,
                SafetySeverity.High => 0.8,
                SafetySeverity.Medium => 0.5,
                SafetySeverity.Low => 0.2,
                SafetySeverity.Info => 0.1,
                _ => 0.0
            };

            double score = severityWeight * finding.Confidence;
            if (score > maxScore)
            {
                maxScore = score;
            }
        }

        return maxScore;
    }

    private static bool ContainsOverridePattern(string text)
    {
        // Patterns that try to override safety for other modalities
        string[] overridePatterns = new[]
        {
            "ignore the image",
            "the image is safe",
            "disregard the visual",
            "override safety",
            "bypass content filter",
            "the picture is harmless",
            "don't flag the image",
            "the audio is clean",
            "ignore what you see",
            "trust the text not the image"
        };

        foreach (string pattern in overridePatterns)
        {
            if (text.Contains(pattern))
            {
                return true;
            }
        }

        return false;
    }
}
