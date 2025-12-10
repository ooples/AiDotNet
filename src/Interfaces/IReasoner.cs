using AiDotNet.Reasoning;
using AiDotNet.Reasoning.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for the main reasoning facade that provides a simple, unified API
/// for solving problems using advanced AI reasoning strategies.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This interface defines what the Reasoner can do.
/// It's the "menu" of reasoning capabilities available to you:
/// - Solve problems with customizable strategies
/// - Get quick answers for simple problems
/// - Do deep analysis for complex problems
/// - Use consensus voting for high-confidence answers
///
/// The actual implementation is in the <see cref="Reasoner{T}"/> class.
/// </para>
/// </remarks>
internal interface IReasoner<T>
{
    /// <summary>
    /// Solves a problem using the specified reasoning mode and configuration.
    /// </summary>
    /// <param name="problem">The problem or question to solve.</param>
    /// <param name="mode">The reasoning mode to use (default: Auto).</param>
    /// <param name="config">Configuration options (default: balanced settings).</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A complete reasoning result with answer, steps, and metrics.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main method for solving problems.
    /// You provide a problem, optionally specify how to solve it, and get back
    /// a complete result with the answer and all the reasoning steps.
    ///
    /// <code>
    /// var result = await reasoner.SolveAsync(
    ///     "If a train travels 60 mph for 2.5 hours, how far does it go?",
    ///     ReasoningMode.ChainOfThought
    /// );
    /// Console.WriteLine(result.FinalAnswer);  // "150 miles"
    /// </code>
    /// </para>
    /// </remarks>
    Task<ReasoningResult<T>> SolveAsync(
        string problem,
        ReasoningMode mode = ReasoningMode.Auto,
        ReasoningConfig? config = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Quickly solves a problem with minimal reasoning overhead.
    /// </summary>
    /// <param name="problem">The problem or question to solve.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>The final answer as a string.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this for simple problems where you just want
    /// a quick answer without detailed reasoning steps. It's faster but less thorough.
    ///
    /// <code>
    /// string answer = await reasoner.QuickSolveAsync("What is 15% of 240?");
    /// Console.WriteLine(answer);  // "36"
    /// </code>
    /// </para>
    /// </remarks>
    Task<string> QuickSolveAsync(
        string problem,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Performs deep, thorough reasoning on a complex problem.
    /// </summary>
    /// <param name="problem">The complex problem to analyze.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A comprehensive reasoning result with extensive exploration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this for complex problems that need careful analysis.
    /// It explores multiple approaches, verifies reasoning, and provides high-confidence answers.
    /// Takes longer but produces more reliable results.
    ///
    /// <code>
    /// var result = await reasoner.DeepSolveAsync(
    ///     "Design an algorithm to find the shortest path in a weighted graph"
    /// );
    /// // Result includes multiple explored approaches
    /// </code>
    /// </para>
    /// </remarks>
    Task<ReasoningResult<T>> DeepSolveAsync(
        string problem,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Solves a problem multiple times and uses consensus to determine the answer.
    /// </summary>
    /// <param name="problem">The problem to solve.</param>
    /// <param name="numAttempts">Number of independent solving attempts (default: 5).</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>The consensus result with voting statistics.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method solves the same problem multiple times
    /// independently and picks the most common answer. It's like asking 5 experts
    /// and going with the majority opinion.
    ///
    /// Great for problems where you need high confidence in the answer.
    ///
    /// <code>
    /// var result = await reasoner.SolveWithConsensusAsync(
    ///     "What is the derivative of x^3?",
    ///     numAttempts: 5
    /// );
    /// // If 4 out of 5 attempts say "3x^2", that's the answer
    /// </code>
    /// </para>
    /// </remarks>
    Task<ReasoningResult<T>> SolveWithConsensusAsync(
        string problem,
        int numAttempts = 5,
        CancellationToken cancellationToken = default);
}
