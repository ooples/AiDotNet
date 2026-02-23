using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Combines multiple toxicity detectors into a weighted ensemble for improved accuracy.
/// </summary>
/// <remarks>
/// <para>
/// Aggregates findings from multiple toxicity detection strategies (rule-based, embedding-based,
/// classifier-based) using configurable weights. The ensemble approach reduces both false positives
/// and false negatives compared to any single detector.
/// </para>
/// <para>
/// <b>For Beginners:</b> Just like a panel of judges gives better verdicts than a single judge,
/// combining multiple toxicity detectors gives more accurate results. This module runs several
/// different detection approaches and combines their opinions.
/// </para>
/// <para>
/// <b>References:</b>
/// - Ensemble methods for robust hate speech detection (ACL 2024)
/// - MetaTox knowledge graph for enhanced LLM toxicity detection (2024, arxiv:2412.15268)
/// - GPT-4o/LLaMA-3 zero-shot and few-shot hate speech detection (2025, arxiv:2506.12744)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EnsembleToxicityDetector<T> : TextSafetyModuleBase<T>
{
    private readonly ITextSafetyModule<T>[] _detectors;
    private readonly double[] _weights;
    private readonly double _threshold;

    /// <inheritdoc />
    public override string ModuleName => "EnsembleToxicityDetector";

    /// <summary>
    /// Initializes a new ensemble toxicity detector with default sub-detectors.
    /// </summary>
    /// <param name="threshold">Ensemble threshold (0-1). Default: 0.5.</param>
    public EnsembleToxicityDetector(double threshold = 0.5)
    {
        if (threshold < 0 || threshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold),
                "Threshold must be between 0 and 1.");
        }

        _threshold = threshold;
        _detectors = new ITextSafetyModule<T>[]
        {
            new RuleBasedToxicityDetector<T>(threshold * 0.8), // More sensitive for rules
            new EmbeddingToxicityDetector<T>(threshold * 0.9),
            new ClassifierToxicityDetector<T>(threshold * 0.9),
        };
        _weights = new[] { 0.3, 0.35, 0.35 }; // Slightly favor ML-based detectors
    }

    /// <summary>
    /// Initializes a new ensemble toxicity detector with custom sub-detectors and weights.
    /// </summary>
    /// <param name="detectors">The sub-detectors to use.</param>
    /// <param name="weights">Weight for each detector. Must sum to ~1.0.</param>
    /// <param name="threshold">Ensemble threshold (0-1). Default: 0.5.</param>
    public EnsembleToxicityDetector(
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
        var allFindings = new List<IReadOnlyList<SafetyFinding>>();
        for (int i = 0; i < _detectors.Length; i++)
        {
            allFindings.Add(_detectors[i].EvaluateText(text));
        }

        // Group findings by category across detectors
        var categoryScores = new Dictionary<SafetyCategory, double>();
        var categorySeverity = new Dictionary<SafetyCategory, SafetySeverity>();

        for (int d = 0; d < allFindings.Count; d++)
        {
            double weight = _weights[d];
            foreach (var finding in allFindings[d])
            {
                if (!categoryScores.ContainsKey(finding.Category))
                {
                    categoryScores[finding.Category] = 0;
                    categorySeverity[finding.Category] = SafetySeverity.Info;
                }

                categoryScores[finding.Category] += finding.Confidence * weight;
                if (finding.Severity > categorySeverity[finding.Category])
                {
                    categorySeverity[finding.Category] = finding.Severity;
                }
            }
        }

        // Emit findings that exceed ensemble threshold
        var results = new List<SafetyFinding>();
        foreach (var (category, score) in categoryScores)
        {
            if (score >= _threshold)
            {
                double clampedScore = Math.Min(1.0, Math.Max(0.0, score));
                results.Add(new SafetyFinding
                {
                    Category = category,
                    Severity = categorySeverity[category],
                    Confidence = clampedScore,
                    Description = $"Ensemble toxicity detection for {category} (score: {clampedScore:F3}, " +
                                  $"{_detectors.Length} detectors).",
                    RecommendedAction = clampedScore >= 0.8 ? SafetyAction.Block : SafetyAction.Warn,
                    SourceModule = ModuleName
                });
            }
        }

        return results;
    }
}
