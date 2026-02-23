using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Combines multiple jailbreak detection strategies into a robust ensemble.
/// </summary>
/// <remarks>
/// <para>
/// Runs pattern-based and semantic jailbreak detectors in parallel and aggregates their
/// findings using weighted voting. A detection from multiple strategies receives a higher
/// confidence score than a single-strategy detection.
/// </para>
/// <para>
/// <b>For Beginners:</b> Attackers constantly invent new ways to bypass AI safety measures.
/// By combining multiple detection approaches — pattern matching, semantic analysis, and
/// encoding detection — this ensemble catches a wider variety of attacks than any single
/// method could alone.
/// </para>
/// <para>
/// <b>References:</b>
/// - GuardReasoner: Reasoning-based explainable guardrails (2025, arxiv:2501.18492)
/// - Qwen3Guard: 85.3% accuracy, robust to prompt variation (Alibaba, 2025, arxiv:2510.14276)
/// - Granite Guardian: 81.0% accuracy with minimal prompt sensitivity (IBM, 2024, arxiv:2412.07724)
/// - LoRA-Guard: Parameter-efficient customizable guardrails (2024, arxiv:2407.02987)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EnsembleJailbreakDetector<T> : TextSafetyModuleBase<T>
{
    private readonly ITextSafetyModule<T>[] _detectors;
    private readonly double[] _weights;
    private readonly double _threshold;

    /// <inheritdoc />
    public override string ModuleName => "EnsembleJailbreakDetector";

    /// <summary>
    /// Initializes a new ensemble jailbreak detector with default sub-detectors.
    /// </summary>
    /// <param name="threshold">Detection threshold (0-1). Default: 0.5.</param>
    /// <param name="sensitivity">
    /// Sensitivity level for the pattern detector (0-1). Default: 0.5.
    /// </param>
    public EnsembleJailbreakDetector(double threshold = 0.5, double sensitivity = 0.5)
    {
        if (threshold < 0 || threshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold),
                "Threshold must be between 0 and 1.");
        }

        _threshold = threshold;
        _detectors = new ITextSafetyModule<T>[]
        {
            new PatternJailbreakDetector<T>(sensitivity),
            new SemanticJailbreakDetector<T>(threshold * 0.9),
        };
        _weights = new[] { 0.5, 0.5 };
    }

    /// <summary>
    /// Initializes a new ensemble jailbreak detector with custom sub-detectors.
    /// </summary>
    /// <param name="detectors">The sub-detectors to use.</param>
    /// <param name="weights">Weight for each detector.</param>
    /// <param name="threshold">Ensemble threshold (0-1). Default: 0.5.</param>
    public EnsembleJailbreakDetector(
        ITextSafetyModule<T>[] detectors,
        double[] weights,
        double threshold = 0.5)
    {
        if (detectors is null) throw new ArgumentNullException(nameof(detectors));
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        if (detectors.Length == 0) throw new ArgumentException("At least one detector is required.", nameof(detectors));
        if (detectors.Length != weights.Length)
        {
            throw new ArgumentException("Number of detectors must match number of weights.");
        }
        if (threshold < 0 || threshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold),
                "Threshold must be between 0 and 1.");
        }
        for (int i = 0; i < detectors.Length; i++)
        {
            if (detectors[i] is null) throw new ArgumentException($"Detector at index {i} is null.", nameof(detectors));
        }
        for (int i = 0; i < weights.Length; i++)
        {
            if (weights[i] < 0) throw new ArgumentException($"Weight at index {i} is negative.", nameof(weights));
        }
        double weightSum = 0;
        for (int i = 0; i < weights.Length; i++) weightSum += weights[i];
        if (weightSum <= 0) throw new ArgumentException("Weights must sum to a positive value.", nameof(weights));

        _threshold = threshold;
        _detectors = detectors;
        // Normalize weights to sum to 1.0
        var normalized = new double[weights.Length];
        for (int i = 0; i < weights.Length; i++) normalized[i] = weights[i] / weightSum;
        _weights = normalized;
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return Array.Empty<SafetyFinding>();
        }

        // Collect findings from all detectors
        int detectorsTriggered = 0;
        double weightedScore = 0;
        double maxConfidence = 0;
        SafetySeverity maxSeverity = SafetySeverity.Info;
        var allDescriptions = new List<string>();

        for (int i = 0; i < _detectors.Length; i++)
        {
            var detectorFindings = _detectors[i].EvaluateText(text);
            var jailbreakFindings = new List<SafetyFinding>();

            foreach (var f in detectorFindings)
            {
                if (f.Category == SafetyCategory.JailbreakAttempt ||
                    f.Category == SafetyCategory.PromptInjection)
                {
                    jailbreakFindings.Add(f);
                }
            }

            if (jailbreakFindings.Count > 0)
            {
                detectorsTriggered++;
                double maxDetectorConfidence = 0;

                foreach (var f in jailbreakFindings)
                {
                    if (f.Confidence > maxDetectorConfidence)
                    {
                        maxDetectorConfidence = f.Confidence;
                    }
                    if (f.Severity > maxSeverity)
                    {
                        maxSeverity = f.Severity;
                    }
                }

                weightedScore += maxDetectorConfidence * _weights[i];
                if (maxDetectorConfidence > maxConfidence)
                {
                    maxConfidence = maxDetectorConfidence;
                }

                allDescriptions.Add($"{_detectors[i].ModuleName}: {maxDetectorConfidence:F3}");
            }
        }

        var results = new List<SafetyFinding>();

        if (weightedScore >= _threshold)
        {
            // Boost score if multiple detectors agree
            double agreementBoost = detectorsTriggered > 1 ? 1.1 : 1.0;
            double finalScore = Math.Min(1.0, weightedScore * agreementBoost);

            results.Add(new SafetyFinding
            {
                Category = SafetyCategory.JailbreakAttempt,
                Severity = maxSeverity,
                Confidence = finalScore,
                Description = $"Ensemble jailbreak detection (score: {finalScore:F3}, " +
                              $"{detectorsTriggered}/{_detectors.Length} detectors triggered). " +
                              $"Details: {string.Join(", ", allDescriptions)}.",
                RecommendedAction = SafetyAction.Block,
                SourceModule = ModuleName
            });
        }

        return results;
    }
}
