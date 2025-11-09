using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Models;

/// <summary>
/// Represents a reasoning step with verification information.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class VerifiedReasoningStep<T>
{
    /// <summary>
    /// The reasoning statement.
    /// </summary>
    public string Statement { get; set; } = string.Empty;

    /// <summary>
    /// Documents supporting this reasoning step.
    /// </summary>
    public List<Document<T>> SupportingDocuments { get; set; } = new List<Document<T>>();

    /// <summary>
    /// Verification score (0-1, higher is better).
    /// </summary>
    public double VerificationScore { get; set; }

    /// <summary>
    /// Whether this step passed verification.
    /// </summary>
    public bool IsVerified { get; set; }

    /// <summary>
    /// Critique feedback from the critic model.
    /// </summary>
    public string CritiqueFeedback { get; set; } = string.Empty;

    /// <summary>
    /// Number of refinement attempts for this step.
    /// </summary>
    public int RefinementAttempts { get; set; }

    /// <summary>
    /// Original statement before refinement (if any).
    /// </summary>
    public string? OriginalStatement { get; set; }
}
