using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Reasoning.Models;

/// <summary>
/// Represents a complete chain of reasoning steps from problem to solution.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring and calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A reasoning chain is like showing your complete work on a problem,
/// from start to finish. Just like in math class where you write out all your steps:
///
/// Problem: "What is 15% of 240?"
/// Step 1: Convert percentage to decimal
/// Step 2: Multiply
/// Step 3: State the answer
///
/// The ReasoningChain class keeps track of all these steps together, along with scores that
/// tell you how confident the AI is about each step. It uses a Vector to store scores efficiently,
/// which is important for machine learning operations.
/// </para>
/// <para><b>Example Usage:</b>
/// <code>
/// var chain = new ReasoningChain&lt;double&gt;
/// {
///     Query = "What is 15% of 240?",
///     Steps = new List&lt;ReasoningStep&lt;double&gt;&gt;
///     {
///         new() { StepNumber = 1, Content = "Convert 15% to 0.15", Score = 1.0 },
///         new() { StepNumber = 2, Content = "Multiply: 0.15 × 240 = 36", Score = 0.95 },
///         new() { StepNumber = 3, Content = "Final answer: 36", Score = 1.0 }
///     }
/// };
///
/// // Get all step scores as a vector
/// var scoreVector = chain.StepScores;
/// Console.WriteLine($"Average confidence: {scoreVector.Mean()}");
/// </code>
/// </para>
/// </remarks>
public class ReasoningChain<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="ReasoningChain{T}"/> class.
    /// </summary>
    public ReasoningChain()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        OverallScore = _numOps.Zero;
    }

    /// <summary>
    /// The original query or problem that this reasoning chain addresses.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the question or problem you're trying to solve.
    /// Everything in the reasoning chain is working toward answering this query.
    /// </para>
    /// </remarks>
    public string Query { get; set; } = string.Empty;

    /// <summary>
    /// The ordered list of reasoning steps in this chain.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is your complete "showing your work" - all the steps
    /// from start to finish, in order. Each step builds on the previous ones.
    /// </para>
    /// </remarks>
    public List<ReasoningStep<T>> Steps { get; set; } = new();

    /// <summary>
    /// Vector of confidence scores for each step, enabling efficient ML operations.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This property provides all the step scores as a Vector,
    /// which is a special data structure optimized for mathematical operations. It's like
    /// having all your test scores in a format where you can easily calculate averages,
    /// find the lowest score, or perform other statistical analyses.
    ///
    /// The Vector is automatically generated from the steps' scores, so you don't need
    /// to manually keep it updated.
    ///
    /// Why use a Vector instead of a regular list?
    /// - Vectors support mathematical operations (mean, standard deviation, etc.)
    /// - They're optimized for machine learning algorithms
    /// - They integrate with other linear algebra operations in the library
    /// </para>
    /// </remarks>
    public Vector<T> StepScores
    {
        get
        {
            if (Steps.Count == 0)
            {
                // Return empty vector if no steps
                return new Vector<T>(0);
            }

            return new Vector<T>(Steps.Select(s => s.Score));
        }
    }

    /// <summary>
    /// The final answer or conclusion from this reasoning chain.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is your final answer - the result of all the reasoning steps.
    /// In a math problem, this would be the number you circle at the end.
    /// </para>
    /// </remarks>
    public string FinalAnswer { get; set; } = string.Empty;

    /// <summary>
    /// Overall confidence score for the entire reasoning chain.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how confident the AI is about the entire solution,
    /// considering all steps together. It's often calculated as the minimum or average of
    /// all step scores, because:
    /// - A chain is only as strong as its weakest link
    /// - Low confidence in any step reduces confidence in the final answer
    ///
    /// For example, if steps have scores [0.9, 0.8, 0.95], the overall confidence might be
    /// 0.8 (the minimum) or 0.88 (the average).
    /// </para>
    /// </remarks>
    public T OverallScore { get; set; }

    /// <summary>
    /// Whether every step in this chain has been verified.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you if all steps have been checked and confirmed.
    /// It's like having a teacher review every line of your work rather than just the final answer.
    ///
    /// Returns true only if ALL steps are verified; if even one step is unverified, returns false.
    /// </para>
    /// </remarks>
    public bool IsFullyVerified => Steps.Count > 0 && Steps.All(s => s.IsVerified);

    /// <summary>
    /// Total number of refinements made across all steps in this chain.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This counts how many corrections were made in total.
    /// A high number might indicate:
    /// - The problem was difficult
    /// - Initial approaches had errors that needed fixing
    /// - The reasoning required multiple iterations to get right
    ///
    /// Lower numbers generally indicate cleaner, more straightforward reasoning.
    /// </para>
    /// </remarks>
    public int TotalRefinements => Steps.Sum(s => s.RefinementCount);

    /// <summary>
    /// When this reasoning chain was started.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Records when the AI started thinking about this problem.
    /// </para>
    /// </remarks>
    public DateTime StartedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// When this reasoning chain was completed.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Records when the AI finished and reached a final answer.
    /// The difference between CompletedAt and StartedAt tells you how long the reasoning took.
    /// </para>
    /// </remarks>
    public DateTime? CompletedAt { get; set; }

    /// <summary>
    /// Total time spent on this reasoning chain.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This calculates how long the reasoning took. If the chain
    /// isn't completed yet, it returns the time elapsed so far.
    /// </para>
    /// </remarks>
    public TimeSpan Duration =>
        CompletedAt.HasValue
            ? CompletedAt.Value - StartedAt
            : DateTime.UtcNow - StartedAt;

    /// <summary>
    /// Additional metadata or context for this reasoning chain.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A flexible storage area for any extra information about
    /// this reasoning chain, such as:
    /// - Which strategy was used (Chain-of-Thought, Tree-of-Thoughts, etc.)
    /// - What tools were used
    /// - Domain-specific information
    /// - References or citations
    /// </para>
    /// </remarks>
    public Dictionary<string, object> Metadata { get; set; } = new();

    /// <summary>
    /// Adds a reasoning step to this chain.
    /// </summary>
    /// <param name="step">The step to add.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This adds a new step to the end of your reasoning chain.
    /// The step number is automatically set to be one more than the current last step.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown if step is null.</exception>
    public void AddStep(ReasoningStep<T> step)
    {
        if (step == null)
            throw new ArgumentNullException(nameof(step));

        step.StepNumber = Steps.Count + 1;
        Steps.Add(step);
    }

    /// <summary>
    /// Gets the minimum confidence score across all steps.
    /// </summary>
    /// <returns>The lowest score from any step, or default if no steps exist.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This finds the weakest link in your reasoning chain -
    /// the step you're least confident about. This is useful because:
    /// - Your overall confidence can't be higher than your least confident step
    /// - It helps identify which steps might need more work or verification
    /// </para>
    /// </remarks>
    public T GetMinimumScore()
    {
        if (Steps.Count == 0)
            return _numOps.Zero;

        return StepScores.Min();
    }

    /// <summary>
    /// Gets the average confidence score across all steps.
    /// </summary>
    /// <returns>The mean score from all steps, or default if no steps exist.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This calculates the average confidence across all steps,
    /// giving you a general sense of how reliable the reasoning is overall.
    /// </para>
    /// </remarks>
    public T GetAverageScore()
    {
        if (Steps.Count == 0)
            return _numOps.Zero;

        return StepScores.Mean();
    }

    /// <summary>
    /// Returns a formatted string representation of the entire reasoning chain.
    /// </summary>
    /// <returns>A multi-line string showing the query, all steps, and final answer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a readable version of the entire reasoning process,
    /// useful for displaying to users or logging for debugging.
    /// </para>
    /// </remarks>
    public override string ToString()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Query: {Query}");
        sb.AppendLine($"\nReasoning Steps ({Steps.Count}):");
        foreach (var step in Steps)
        {
            var verified = step.IsVerified ? "✓" : " ";
            sb.AppendLine($"  [{verified}] {step}");
        }
        sb.AppendLine($"\nFinal Answer: {FinalAnswer}");
        sb.AppendLine($"Overall Score: {OverallScore}");
        sb.AppendLine($"Duration: {Duration.TotalSeconds:F2}s");
        return sb.ToString();
    }
}
