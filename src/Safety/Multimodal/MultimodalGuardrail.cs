using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Multimodal;

/// <summary>
/// Unified guardrail for vision-language models (VLMs) and multimodal AI systems that
/// validates both text and image content together.
/// </summary>
/// <remarks>
/// <para>
/// Provides input/output guardrailing for multimodal systems. For text-only content,
/// delegates to configured text safety modules. For image-only content, delegates to
/// configured image safety modules. For combined text+image content, additionally checks
/// for cross-modal attacks where individually safe content becomes harmful when combined.
/// </para>
/// <para>
/// <b>For Beginners:</b> When AI systems accept both text and images (like "describe this image"),
/// attackers can exploit the gap between modalities. An image might be harmless by itself, and a
/// prompt might be harmless by itself, but together they could trick the AI. This guardrail
/// checks the combination to catch such attacks.
/// </para>
/// <para>
/// <b>References:</b>
/// - Visual prompt injection attacks on GPT-4V (2024)
/// - MM-SafetyBench: Multimodal safety benchmark (2024)
/// - Cross-modal jailbreak attacks on multimodal LLMs (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MultimodalGuardrail<T> : MultimodalSafetyModuleBase<T>
{
    private readonly IReadOnlyList<ITextSafetyModule<T>> _textModules;
    private readonly IReadOnlyList<IImageSafetyModule<T>> _imageModules;
    private readonly double _crossModalThreshold;

    // Terms that in text context may indicate cross-modal attack vectors
    private static readonly HashSet<string> SuspiciousTextImagePatterns = new(StringComparer.OrdinalIgnoreCase)
    {
        "ignore previous", "ignore above", "disregard", "override instructions",
        "new instructions", "system prompt", "you are now", "act as",
        "forget everything", "ignore all", "bypass", "jailbreak"
    };

    /// <inheritdoc />
    public override string ModuleName => "MultimodalGuardrail";

    /// <summary>
    /// Initializes a new multimodal guardrail.
    /// </summary>
    /// <param name="textModules">Text safety modules for evaluating text portions.</param>
    /// <param name="imageModules">Image safety modules for evaluating image portions.</param>
    /// <param name="crossModalThreshold">
    /// Threshold for cross-modal attack detection (0.0-1.0). Default: 0.5.
    /// Lower values are more sensitive.
    /// </param>
    public MultimodalGuardrail(
        IReadOnlyList<ITextSafetyModule<T>>? textModules = null,
        IReadOnlyList<IImageSafetyModule<T>>? imageModules = null,
        double crossModalThreshold = 0.5)
    {
        _textModules = textModules ?? Array.Empty<ITextSafetyModule<T>>();
        _imageModules = imageModules ?? Array.Empty<IImageSafetyModule<T>>();
        _crossModalThreshold = crossModalThreshold;
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateTextImage(string text, Tensor<T> image)
    {
        var findings = new List<SafetyFinding>();

        // Phase 1: Evaluate text through text safety modules
        if (!string.IsNullOrWhiteSpace(text))
        {
            foreach (var module in _textModules)
            {
                var textFindings = module.EvaluateText(text);
                findings.AddRange(textFindings);
            }
        }

        // Phase 2: Evaluate image through image safety modules
        if (image.Data.Length > 0)
        {
            foreach (var module in _imageModules)
            {
                var imageFindings = module.EvaluateImage(image);
                findings.AddRange(imageFindings);
            }
        }

        // Phase 3: Cross-modal attack detection
        if (!string.IsNullOrWhiteSpace(text) && image.Data.Length > 0)
        {
            double crossModalScore = DetectCrossModalAttack(text, image);
            if (crossModalScore >= _crossModalThreshold)
            {
                findings.Add(new SafetyFinding
                {
                    Category = SafetyCategory.PromptInjection,
                    Severity = crossModalScore >= 0.8 ? SafetySeverity.Critical : SafetySeverity.High,
                    Confidence = crossModalScore,
                    Description = $"Cross-modal attack pattern detected (score: {crossModalScore:F3}). " +
                                  "Text and image combination may constitute a visual prompt injection " +
                                  "or cross-modal jailbreak attempt.",
                    RecommendedAction = SafetyAction.Block,
                    SourceModule = ModuleName
                });
            }

            // Check for text-in-image OCR attack vectors
            double textInImageScore = DetectTextInImageAttack(text, image);
            if (textInImageScore >= _crossModalThreshold)
            {
                findings.Add(new SafetyFinding
                {
                    Category = SafetyCategory.PromptInjection,
                    Severity = SafetySeverity.High,
                    Confidence = textInImageScore,
                    Description = $"Possible text-in-image attack detected (score: {textInImageScore:F3}). " +
                                  "Image may contain embedded text designed to manipulate model behavior.",
                    RecommendedAction = SafetyAction.Warn,
                    SourceModule = ModuleName
                });
            }
        }

        return findings;
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateTextAudio(string text, Vector<T> audio, int sampleRate)
    {
        var findings = new List<SafetyFinding>();

        // Evaluate text through text safety modules
        if (!string.IsNullOrWhiteSpace(text))
        {
            foreach (var module in _textModules)
            {
                findings.AddRange(module.EvaluateText(text));
            }
        }

        return findings;
    }

    private double DetectCrossModalAttack(string text, Tensor<T> image)
    {
        // Check text for patterns associated with visual prompt injection
        string lower = text.ToLowerInvariant();
        int suspiciousCount = 0;

        foreach (string pattern in SuspiciousTextImagePatterns)
        {
            if (lower.Contains(pattern, StringComparison.OrdinalIgnoreCase))
            {
                suspiciousCount++;
            }
        }

        if (suspiciousCount == 0) return 0;

        // Score based on number of suspicious patterns found
        double textScore = Math.Min(1.0, suspiciousCount / 3.0);

        // Check image for unusually uniform regions (possible text overlay areas)
        double imageUniformity = ComputeImageUniformity(image);

        // High text suspicion + uniform image regions = likely visual prompt injection
        return textScore * 0.7 + imageUniformity * 0.3;
    }

    private double DetectTextInImageAttack(string text, Tensor<T> image)
    {
        var span = image.Data.Span;
        if (span.Length < 64) return 0;

        // Detect high-contrast regions that may contain embedded text
        int sampleSize = Math.Min(span.Length, 4096);
        int highContrastTransitions = 0;
        int totalChecked = 0;

        for (int i = 1; i < sampleSize; i++)
        {
            double prev = NumOps.ToDouble(span[i - 1]);
            double curr = NumOps.ToDouble(span[i]);
            double diff = Math.Abs(curr - prev);

            if (prev > 1.0) prev /= 255.0;
            if (curr > 1.0) curr /= 255.0;
            diff = Math.Abs(curr - prev);

            if (diff > 0.5) highContrastTransitions++;
            totalChecked++;
        }

        if (totalChecked < 32) return 0;

        // High-contrast transitions are typical of text rendered in images
        double transitionRate = (double)highContrastTransitions / totalChecked;

        // Natural images: ~5-15% high-contrast transitions
        // Text in images: >20% high-contrast transitions
        if (transitionRate < 0.15) return 0;
        return Math.Min(1.0, (transitionRate - 0.15) / 0.25);
    }

    private double ComputeImageUniformity(Tensor<T> image)
    {
        var span = image.Data.Span;
        if (span.Length < 64) return 0;

        // Compute variance of a sample to detect uniform regions
        int sampleSize = Math.Min(span.Length, 2048);
        double sum = 0, sumSq = 0;

        for (int i = 0; i < sampleSize; i++)
        {
            double val = NumOps.ToDouble(span[i]);
            if (val > 1.0) val /= 255.0;
            sum += val;
            sumSq += val * val;
        }

        double mean = sum / sampleSize;
        double variance = (sumSq / sampleSize) - (mean * mean);

        // Very low variance = very uniform = possible text overlay background
        if (variance < 0.01) return Math.Min(1.0, (0.01 - variance) / 0.01);
        return 0;
    }
}
