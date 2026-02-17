using AiDotNet.Enums;

namespace AiDotNet.Safety;

/// <summary>
/// Unified safety report aggregating findings from all safety modules in the pipeline.
/// </summary>
/// <remarks>
/// <para>
/// A SafetyReport is the single output object from the safety pipeline. It aggregates
/// all findings from every safety module that ran, provides an overall safety verdict,
/// and recommends the strictest action needed.
/// </para>
/// <para>
/// <b>For Beginners:</b> After the safety system finishes checking your content,
/// it produces this report summarizing everything it found. The key properties are:
/// - <see cref="IsSafe"/>: Quick yes/no answer
/// - <see cref="OverallAction"/>: What the system recommends doing
/// - <see cref="Findings"/>: Detailed list of every issue found
/// </para>
/// </remarks>
public class SafetyReport
{
    /// <summary>
    /// Gets whether the content passed all safety checks.
    /// </summary>
    /// <remarks>
    /// True only if no findings have severity Medium or above and no Block/Quarantine actions.
    /// </remarks>
    public bool IsSafe { get; init; }

    /// <summary>
    /// Gets the overall recommended action (the strictest action from all findings).
    /// </summary>
    public SafetyAction OverallAction { get; init; } = SafetyAction.Allow;

    /// <summary>
    /// Gets the highest severity found across all findings.
    /// </summary>
    public SafetySeverity HighestSeverity { get; init; } = SafetySeverity.Info;

    /// <summary>
    /// Gets the overall safety score (0.0 = completely unsafe, 1.0 = completely safe).
    /// </summary>
    /// <remarks>
    /// Computed as the minimum safety score across all modules that contributed findings.
    /// If no findings exist, the score is 1.0.
    /// </remarks>
    public double OverallScore { get; init; } = 1.0;

    /// <summary>
    /// Gets the list of all safety findings from all modules.
    /// </summary>
    public IReadOnlyList<SafetyFinding> Findings { get; init; } = Array.Empty<SafetyFinding>();

    /// <summary>
    /// Gets the categories of harmful content detected.
    /// </summary>
    public IReadOnlyList<SafetyCategory> DetectedCategories { get; init; } = Array.Empty<SafetyCategory>();

    /// <summary>
    /// Gets the duration of the safety evaluation in milliseconds.
    /// </summary>
    public double EvaluationTimeMs { get; init; }

    /// <summary>
    /// Gets the names of all safety modules that were executed.
    /// </summary>
    public IReadOnlyList<string> ModulesExecuted { get; init; } = Array.Empty<string>();

    /// <summary>
    /// Creates a safe (no findings) report.
    /// </summary>
    /// <param name="modulesExecuted">Names of the modules that ran.</param>
    /// <param name="evaluationTimeMs">Time taken in milliseconds.</param>
    /// <returns>A SafetyReport indicating the content is safe.</returns>
    public static SafetyReport Safe(IReadOnlyList<string> modulesExecuted, double evaluationTimeMs = 0)
    {
        return new SafetyReport
        {
            IsSafe = true,
            OverallAction = SafetyAction.Allow,
            HighestSeverity = SafetySeverity.Info,
            OverallScore = 1.0,
            Findings = Array.Empty<SafetyFinding>(),
            DetectedCategories = Array.Empty<SafetyCategory>(),
            EvaluationTimeMs = evaluationTimeMs,
            ModulesExecuted = modulesExecuted
        };
    }

    /// <summary>
    /// Creates a report from a list of findings.
    /// </summary>
    /// <param name="findings">All findings from all modules.</param>
    /// <param name="modulesExecuted">Names of the modules that ran.</param>
    /// <param name="evaluationTimeMs">Time taken in milliseconds.</param>
    /// <returns>A SafetyReport summarizing the findings.</returns>
    public static SafetyReport FromFindings(
        IReadOnlyList<SafetyFinding> findings,
        IReadOnlyList<string> modulesExecuted,
        double evaluationTimeMs = 0)
    {
        if (findings.Count == 0)
        {
            return Safe(modulesExecuted, evaluationTimeMs);
        }

        var highestSeverity = SafetySeverity.Info;
        var strictestAction = SafetyAction.Allow;
        var categories = new HashSet<SafetyCategory>();
        double minConfidenceWeightedScore = 1.0;

        foreach (var finding in findings)
        {
            if (finding.Severity > highestSeverity)
            {
                highestSeverity = finding.Severity;
            }

            if (finding.RecommendedAction > strictestAction)
            {
                strictestAction = finding.RecommendedAction;
            }

            categories.Add(finding.Category);

            // Safety score decreases based on severity and confidence
            double severityWeight = finding.Severity switch
            {
                SafetySeverity.Critical => 0.0,
                SafetySeverity.High => 0.2,
                SafetySeverity.Medium => 0.5,
                SafetySeverity.Low => 0.8,
                _ => 1.0
            };
            double weightedScore = 1.0 - (1.0 - severityWeight) * finding.Confidence;
            if (weightedScore < minConfidenceWeightedScore)
            {
                minConfidenceWeightedScore = weightedScore;
            }
        }

        bool isSafe = highestSeverity < SafetySeverity.Medium
                       && strictestAction < SafetyAction.Block;

        return new SafetyReport
        {
            IsSafe = isSafe,
            OverallAction = strictestAction,
            HighestSeverity = highestSeverity,
            OverallScore = minConfidenceWeightedScore,
            Findings = findings,
            DetectedCategories = categories.ToArray(),
            EvaluationTimeMs = evaluationTimeMs,
            ModulesExecuted = modulesExecuted
        };
    }
}
