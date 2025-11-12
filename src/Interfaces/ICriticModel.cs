using AiDotNet.Reasoning.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for critic models that evaluate and provide feedback on reasoning steps.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A critic model is like a teacher reviewing your work. It looks at
/// each reasoning step and evaluates:
/// - Is this step logical and well-reasoned?
/// - Is it supported by evidence?
/// - Does it actually help solve the problem?
/// - Are there any errors or weaknesses?
///
/// The critic provides both a score (how good is this step?) and feedback (how to improve it).
///
/// This is similar to peer review in academia or code review in software development - having
/// someone else check your work catches errors you might miss.
///
/// Critic models are essential for:
/// - Verified reasoning (like DeepSeek-R1's approach)
/// - Self-refinement loops
/// - Building high-confidence solutions
/// - Catching hallucinations or logical errors
/// </para>
/// </remarks>
public interface ICriticModel<T>
{
    /// <summary>
    /// Critiques a reasoning step, providing a score and feedback.
    /// </summary>
    /// <param name="step">The reasoning step to critique.</param>
    /// <param name="context">Context including the query and previous steps.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Critique result with score and feedback.</returns>
    Task<CritiqueResult<T>> CritiqueStepAsync(
        ReasoningStep<T> step,
        ReasoningContext context,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Critiques an entire reasoning chain for overall coherence and quality.
    /// </summary>
    /// <param name="chain">The complete reasoning chain.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Critique result for the chain.</returns>
    Task<CritiqueResult<T>> CritiqueChainAsync(
        ReasoningChain<T> chain,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Context information for critiquing reasoning steps.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This provides the critic with background information needed
/// to properly evaluate a reasoning step, like giving a teacher the full assignment when
/// grading one answer.
/// </para>
/// </remarks>
public class ReasoningContext
{
    /// <summary>
    /// The original query or problem being solved.
    /// </summary>
    public string Query { get; set; } = string.Empty;

    /// <summary>
    /// Previous reasoning steps that provide context.
    /// </summary>
    public List<string> PreviousSteps { get; set; } = new();

    /// <summary>
    /// Supporting evidence or documents if available.
    /// </summary>
    public List<string> SupportingEvidence { get; set; } = new();

    /// <summary>
    /// Domain or subject area (e.g., "mathematics", "code", "science").
    /// </summary>
    public string Domain { get; set; } = string.Empty;
}

/// <summary>
/// Result of critiquing a reasoning step or chain.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This is like getting your homework back with a grade and comments.
/// The score tells you how well you did, and the feedback explains what was good or bad.
/// </para>
/// </remarks>
public class CritiqueResult<T>
{
    /// <summary>
    /// Quality score for the reasoning (typically 0.0 to 1.0).
    /// </summary>
    public T Score { get; set; } = default!;

    /// <summary>
    /// Detailed feedback explaining the score.
    /// </summary>
    public string Feedback { get; set; } = string.Empty;

    /// <summary>
    /// Specific strengths identified in the reasoning.
    /// </summary>
    public List<string> Strengths { get; set; } = new();

    /// <summary>
    /// Specific weaknesses or areas for improvement.
    /// </summary>
    public List<string> Weaknesses { get; set; } = new();

    /// <summary>
    /// Suggestions for how to improve this reasoning.
    /// </summary>
    public List<string> Suggestions { get; set; } = new();

    /// <summary>
    /// Whether this reasoning step/chain passes the minimum quality threshold.
    /// </summary>
    public bool PassesThreshold { get; set; }

    public override string ToString() =>
        $"Critique Score: {Score} ({(PassesThreshold ? "PASS" : "FAIL")})\n{Feedback}";
}
