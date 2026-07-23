using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Models;

/// <summary>
/// Result of verified reasoning retrieval.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class VerifiedReasoningResult<T>
{
    /// <summary>
    /// Retrieved documents from all verified steps.
    /// </summary>
    public IEnumerable<Document<T>> Documents { get; set; } = new List<Document<T>>();

    /// <summary>
    /// List of verified reasoning steps.
    /// </summary>
    public IReadOnlyList<VerifiedReasoningStep<T>> VerifiedSteps { get; set; } = new List<VerifiedReasoningStep<T>>();

    /// <summary>
    /// Average verification score across all steps.
    /// </summary>
    public double AverageVerificationScore { get; set; }

    /// <summary>
    /// Number of steps that required refinement.
    /// </summary>
    public int RefinedStepsCount { get; set; }
}
