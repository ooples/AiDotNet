using AiDotNet.LinearAlgebra;

namespace AiDotNet.Reasoning.Models;

/// <summary>
/// Represents the complete result of a reasoning process, including the answer, reasoning chain, and performance metrics.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring and calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Think of ReasoningResult as the complete package you get back after the AI
/// solves a problem. It's like when you finish a homework problem and you have:
/// - The final answer
/// - All your work showing how you got there
/// - Notes about which parts you checked or corrected
/// - How long it took you
/// - How confident you are about the answer
///
/// This class bundles all of that information together in one place, making it easy to work with
/// the results of reasoning.
/// </para>
/// <para><b>Example Usage:</b>
/// <code>
/// // After reasoning completes
/// ReasoningResult&lt;double&gt; result = await strategy.ReasonAsync("What is 15% of 240?");
///
/// // Access the answer
/// Console.WriteLine($"Answer: {result.FinalAnswer}");  // "36"
///
/// // Check confidence
/// Console.WriteLine($"Confidence: {result.OverallConfidence}");  // 0.95
///
/// // See how long it took
/// Console.WriteLine($"Took {result.TotalDuration.TotalSeconds} seconds");
///
/// // Review the reasoning steps
/// foreach (var step in result.ReasoningChain.Steps)
/// {
///     Console.WriteLine($"  - {step.Content}");
/// }
///
/// // Get performance metrics
/// Console.WriteLine($"Used {result.Metrics["llm_calls"]} AI model calls");
/// </code>
/// </para>
/// </remarks>
public class ReasoningResult<T>
{
    /// <summary>
    /// The final answer or solution from the reasoning process.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the bottom-line answer to the question.
    /// For "What is 15% of 240?", this would be "36".
    /// For "How can we reduce emissions?", this might be a detailed explanation.
    /// </para>
    /// </remarks>
    public string FinalAnswer { get; set; } = string.Empty;

    /// <summary>
    /// The complete chain of reasoning steps that led to the final answer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This contains all the "showing your work" - every step
    /// the AI took to reach the final answer. It's important for:
    /// - Understanding how the answer was derived
    /// - Identifying any logical errors
    /// - Building trust in the result
    /// - Learning from the reasoning process
    /// </para>
    /// </remarks>
    public ReasoningChain<T> ReasoningChain { get; set; } = new();

    /// <summary>
    /// Alternative reasoning chains that were explored (for strategies like Tree-of-Thoughts).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some reasoning strategies explore multiple different approaches
    /// to solving a problem. This stores all the alternative solution paths that were considered,
    /// not just the one that was ultimately chosen.
    ///
    /// Think of it like showing multiple ways to solve a math problem - even if you pick one
    /// as your final answer, it's useful to see the other approaches you considered.
    /// </para>
    /// </remarks>
    public List<ReasoningChain<T>> AlternativeChains { get; set; } = new();

    /// <summary>
    /// Overall confidence score for the final answer (0.0 to 1.0).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a single number that represents how confident the AI
    /// is about the final answer, considering everything:
    /// - Confidence in each reasoning step
    /// - Verification results
    /// - Whether multiple approaches agreed
    ///
    /// Think of it as a percentage: 0.95 = 95% confident, 0.50 = 50% confident (uncertain).
    /// Higher values mean you can trust the answer more.
    /// </para>
    /// </remarks>
    public T OverallConfidence { get; set; } = default!;

    /// <summary>
    /// Vector of confidence scores across all attempted reasoning paths.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If multiple reasoning chains were explored, this Vector contains
    /// the confidence score for each one. It's useful for:
    /// - Comparing different approaches
    /// - Calculating statistics (mean, variance of confidence)
    /// - Understanding the distribution of solution quality
    ///
    /// For example, if you tried 5 different approaches and got confidence scores of
    /// [0.9, 0.85, 0.92, 0.75, 0.88], this vector would contain those values.
    /// </para>
    /// </remarks>
    public Vector<T>? ConfidenceScores { get; set; }

    /// <summary>
    /// The strategy that was used for reasoning (e.g., "Chain-of-Thought", "Tree-of-Thoughts").
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This records which reasoning approach was used to solve the problem.
    /// It's useful for:
    /// - Understanding how the solution was generated
    /// - Comparing results from different strategies
    /// - Debugging or analyzing performance
    /// </para>
    /// </remarks>
    public string StrategyUsed { get; set; } = string.Empty;

    /// <summary>
    /// Whether the reasoning process completed successfully.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you if the reasoning finished properly or if something
    /// went wrong. Reasons for failure might include:
    /// - Timeout (took too long)
    /// - No valid reasoning path found
    /// - Critical error during processing
    ///
    /// Always check this before trusting the FinalAnswer!
    /// </para>
    /// </remarks>
    public bool Success { get; set; } = true;

    /// <summary>
    /// Error message if the reasoning failed (null if successful).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If Success is false, this explains what went wrong.
    /// For example:
    /// - "Reasoning timeout after 60 seconds"
    /// - "Unable to find a valid solution path"
    /// - "Verification failed for all reasoning attempts"
    /// </para>
    /// </remarks>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Total time spent on the reasoning process.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How long it took from start to finish, including:
    /// - Generating reasoning steps
    /// - Verifying steps
    /// - Refining incorrect steps
    /// - Exploring alternative paths
    ///
    /// Useful for performance monitoring and optimization.
    /// </para>
    /// </remarks>
    public TimeSpan TotalDuration { get; set; }

    /// <summary>
    /// Verification results and feedback from critic models.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If steps were verified by "critic" models (AI reviewers),
    /// this contains their feedback. Each entry might say things like:
    /// - "Step 2: Logic is sound, calculation verified"
    /// - "Step 4: Needs more justification for this claim"
    ///
    /// It's like teacher comments on your homework.
    /// </para>
    /// </remarks>
    public List<string> VerificationFeedback { get; set; } = new();

    /// <summary>
    /// Tools or external resources that were used during reasoning.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Records which external tools were called during reasoning,
    /// such as:
    /// - "Calculator" - For mathematical calculations
    /// - "PythonInterpreter" - For running code
    /// - "WebSearch" - For looking up information
    ///
    /// Helps track what resources were needed to solve the problem.
    /// </para>
    /// </remarks>
    public List<string> ToolsUsed { get; set; } = new();

    /// <summary>
    /// Performance metrics and statistics about the reasoning process.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This dictionary stores various measurements about how the
    /// reasoning process performed, such as:
    /// - "llm_calls": Number of times the AI model was queried
    /// - "tokens_used": Total tokens consumed
    /// - "verification_checks": Number of verification steps
    /// - "refinement_count": Total refinements made
    /// - "nodes_explored": For tree-based strategies
    ///
    /// Useful for analyzing performance, cost, and optimization opportunities.
    /// </para>
    /// </remarks>
    public Dictionary<string, object> Metrics { get; set; } = new();

    /// <summary>
    /// Additional metadata or context about this reasoning result.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A flexible storage area for any extra information that might
    /// be useful, such as:
    /// - Problem domain (math, code, science, etc.)
    /// - User preferences or constraints
    /// - Session or request IDs
    /// - Custom application-specific data
    /// </para>
    /// </remarks>
    public Dictionary<string, object> Metadata { get; set; } = new();

    /// <summary>
    /// Creates a summary string of the reasoning result.
    /// </summary>
    /// <returns>A formatted string with key information about the result.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a human-readable summary of the result,
    /// useful for logging, debugging, or displaying to users.
    /// </para>
    /// </remarks>
    public string GetSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Reasoning Result ({StrategyUsed})");
        sb.AppendLine($"Success: {Success}");
        if (!Success && ErrorMessage != null)
        {
            sb.AppendLine($"Error: {ErrorMessage}");
        }
        sb.AppendLine($"Final Answer: {FinalAnswer}");
        sb.AppendLine($"Confidence: {OverallConfidence}");
        sb.AppendLine($"Steps: {ReasoningChain.Steps.Count}");
        sb.AppendLine($"Duration: {TotalDuration.TotalSeconds:F2}s");
        if (ToolsUsed.Count > 0)
        {
            sb.AppendLine($"Tools Used: {string.Join(", ", ToolsUsed)}");
        }
        return sb.ToString();
    }

    /// <summary>
    /// Returns a string representation of this reasoning result.
    /// </summary>
    /// <returns>A formatted summary string.</returns>
    public override string ToString() => GetSummary();
}
