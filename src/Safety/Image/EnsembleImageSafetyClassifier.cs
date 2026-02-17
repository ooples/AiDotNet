using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Image;

/// <summary>
/// Combines multiple image safety classifiers into a weighted ensemble for robust detection.
/// </summary>
/// <remarks>
/// <para>
/// Runs multiple image classifiers (CLIP, ViT, SceneGraph) and aggregates their findings
/// using weighted voting. When multiple classifiers agree on a finding, the confidence is
/// boosted. This provides defense-in-depth: each classifier catches different types of
/// unsafe content, and the ensemble's combined coverage is greater than any single classifier.
/// </para>
/// <para>
/// <b>For Beginners:</b> Just like getting multiple doctors' opinions leads to a better
/// diagnosis, running multiple safety classifiers and combining their results gives more
/// reliable detection. If two out of three classifiers flag an image as unsafe, we can be
/// more confident in the detection.
/// </para>
/// <para>
/// <b>References:</b>
/// - UnsafeBench: Ensemble approaches improve F1 (2024, arxiv:2405.03486)
/// - Multi-model safety evaluation for VLMs (2025, arxiv:2512.06589)
/// - Ensemble methods for robust content moderation (Survey, 2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EnsembleImageSafetyClassifier<T> : ImageSafetyModuleBase<T>
{
    private readonly IImageSafetyModule<T>[] _classifiers;
    private readonly double[] _weights;
    private readonly double _threshold;

    /// <inheritdoc />
    public override string ModuleName => "EnsembleImageSafetyClassifier";

    /// <summary>
    /// Initializes a new ensemble image safety classifier with default sub-classifiers.
    /// </summary>
    /// <param name="threshold">Ensemble threshold (0-1). Default: 0.6.</param>
    public EnsembleImageSafetyClassifier(double threshold = 0.6)
    {
        if (threshold < 0 || threshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold),
                "Threshold must be between 0 and 1.");
        }

        _threshold = threshold;
        _classifiers = new IImageSafetyModule<T>[]
        {
            new CLIPImageSafetyClassifier<T>(threshold * 0.9, threshold * 0.9),
            new ViTImageSafetyClassifier<T>(threshold: threshold * 0.9),
            new SceneGraphSafetyClassifier<T>(threshold: threshold * 0.9)
        };
        _weights = new[] { 0.40, 0.35, 0.25 };
    }

    /// <summary>
    /// Initializes a new ensemble image safety classifier with custom sub-classifiers.
    /// </summary>
    /// <param name="classifiers">The image classifiers to combine.</param>
    /// <param name="weights">Weight for each classifier (must sum to ~1.0).</param>
    /// <param name="threshold">Ensemble threshold (0-1). Default: 0.6.</param>
    public EnsembleImageSafetyClassifier(
        IImageSafetyModule<T>[] classifiers,
        double[] weights,
        double threshold = 0.6)
    {
        if (classifiers is null) throw new ArgumentNullException(nameof(classifiers));
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        if (classifiers.Length == 0) throw new ArgumentException("At least one classifier is required.", nameof(classifiers));
        if (classifiers.Length != weights.Length)
        {
            throw new ArgumentException("Number of classifiers must match number of weights.");
        }
        if (threshold < 0 || threshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold),
                "Threshold must be between 0 and 1.");
        }
        for (int i = 0; i < classifiers.Length; i++)
        {
            if (classifiers[i] is null) throw new ArgumentException($"Classifier at index {i} is null.", nameof(classifiers));
        }

        _classifiers = classifiers;
        _weights = weights;
        _threshold = threshold;
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateImage(Tensor<T> image)
    {
        if (image.Data.Span.Length == 0)
        {
            return Array.Empty<SafetyFinding>();
        }

        // Collect findings from all classifiers
        var allFindings = new List<(SafetyFinding finding, int classifierIndex)>();
        for (int i = 0; i < _classifiers.Length; i++)
        {
            var classifierFindings = _classifiers[i].EvaluateImage(image);
            foreach (var f in classifierFindings)
            {
                allFindings.Add((f, i));
            }
        }

        if (allFindings.Count == 0)
        {
            return Array.Empty<SafetyFinding>();
        }

        // Group findings by category
        var categoryGroups = new Dictionary<SafetyCategory, List<(SafetyFinding finding, int classifierIndex)>>();
        foreach (var (finding, idx) in allFindings)
        {
            if (!categoryGroups.TryGetValue(finding.Category, out var group))
            {
                group = new List<(SafetyFinding, int)>();
                categoryGroups[finding.Category] = group;
            }
            group.Add((finding, idx));
        }

        var results = new List<SafetyFinding>();

        foreach (var (category, group) in categoryGroups)
        {
            // Compute weighted ensemble score
            double weightedScore = 0;
            double maxConfidence = 0;
            SafetySeverity maxSeverity = SafetySeverity.Info;
            var detectorNames = new List<string>();

            // Track which classifiers contributed
            var seenClassifiers = new HashSet<int>();
            foreach (var (finding, classifierIdx) in group)
            {
                if (!seenClassifiers.Contains(classifierIdx))
                {
                    weightedScore += finding.Confidence * _weights[classifierIdx];
                    seenClassifiers.Add(classifierIdx);
                }

                if (finding.Confidence > maxConfidence) maxConfidence = finding.Confidence;
                if (finding.Severity > maxSeverity) maxSeverity = finding.Severity;
                detectorNames.Add($"{_classifiers[classifierIdx].ModuleName}:{finding.Confidence:F2}");
            }

            // Agreement boost: multiple classifiers detecting same category
            double agreementBoost = seenClassifiers.Count > 1 ? 1.0 + (seenClassifiers.Count - 1) * 0.1 : 1.0;
            double finalScore = Math.Min(1.0, weightedScore * agreementBoost);

            if (finalScore >= _threshold)
            {
                results.Add(new SafetyFinding
                {
                    Category = category,
                    Severity = maxSeverity,
                    Confidence = finalScore,
                    Description = $"Ensemble image classification: {category} detected (score: {finalScore:F3}, " +
                                  $"{seenClassifiers.Count}/{_classifiers.Length} classifiers triggered). " +
                                  $"Details: {string.Join(", ", detectorNames)}.",
                    RecommendedAction = finalScore >= 0.8 ? SafetyAction.Block : SafetyAction.Warn,
                    SourceModule = ModuleName
                });
            }
        }

        return results;
    }
}
