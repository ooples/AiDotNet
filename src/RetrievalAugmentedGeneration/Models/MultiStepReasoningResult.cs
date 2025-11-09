using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Models;

/// <summary>
/// Result of multi-step reasoning retrieval.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class MultiStepReasoningResult<T>
{
    /// <summary>
    /// All documents retrieved across all steps.
    /// </summary>
    public IEnumerable<Document<T>> Documents { get; set; } = new List<Document<T>>();

    /// <summary>
    /// Detailed results from each reasoning step.
    /// </summary>
    public IReadOnlyList<ReasoningStepResult<T>> StepResults { get; set; } = new List<ReasoningStepResult<T>>();

    /// <summary>
    /// Trace of the reasoning progression.
    /// </summary>
    public string ReasoningTrace { get; set; } = string.Empty;

    /// <summary>
    /// Total number of steps executed.
    /// </summary>
    public int TotalSteps { get; set; }

    /// <summary>
    /// Whether the reasoning converged to a solution.
    /// </summary>
    public bool Converged { get; set; }
}
