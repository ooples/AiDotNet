using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Models;

/// <summary>
/// Represents a single reasoning step in the multi-step process.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class ReasoningStepResult<T>
{
    /// <summary>
    /// The query or reasoning focus for this step.
    /// </summary>
    public string StepQuery { get; set; } = string.Empty;

    /// <summary>
    /// Documents retrieved in this step.
    /// </summary>
    public List<Document<T>> Documents { get; set; } = new List<Document<T>>();

    /// <summary>
    /// Summary of findings from this step.
    /// </summary>
    public string StepSummary { get; set; } = string.Empty;

    /// <summary>
    /// Whether this step yielded useful information.
    /// </summary>
    public bool IsSuccessful { get; set; }

    /// <summary>
    /// The step number in the sequence.
    /// </summary>
    public int StepNumber { get; set; }
}
