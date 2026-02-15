using AiDotNet.Reasoning.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for reasoning strategies that solve problems through structured thinking.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations and scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A reasoning strategy is like a specific approach or method for solving problems.
/// Just like you might use different strategies to solve math problems (working backwards, drawing diagrams,
/// breaking into steps), AI systems can use different reasoning strategies like Chain-of-Thought,
/// Tree-of-Thoughts, or Self-Consistency.
///
/// This interface defines what every reasoning strategy must be able to do:
/// - Accept a problem or query
/// - Apply its specific reasoning approach
/// - Return a structured result with the answer and reasoning trace
///
/// Think of it like different cooking methods (baking, frying, steaming) - they're all ways to prepare
/// food, but each has its own process. Similarly, different reasoning strategies all aim to solve problems,
/// but each uses a different approach.
/// </para>
/// <para><b>Example Usage:</b>
/// <code>
/// // Use Chain-of-Thought strategy for step-by-step reasoning
/// IReasoningStrategy&lt;double&gt; cotStrategy = new ChainOfThoughtStrategy&lt;double&gt;(chatModel);
/// var result = await cotStrategy.ReasonAsync("What is 15% of 240?");
/// Console.WriteLine(result.FinalAnswer); // "36"
/// Console.WriteLine(result.ReasoningChain); // Shows step-by-step work
///
/// // Use Tree-of-Thoughts for exploring multiple paths
/// IReasoningStrategy&lt;double&gt; totStrategy = new TreeOfThoughtsStrategy&lt;double&gt;(chatModel);
/// var result2 = await totStrategy.ReasonAsync("How can we reduce carbon emissions?");
/// </code>
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("ReasoningStrategy")]
public interface IReasoningStrategy<T>
{
    /// <summary>
    /// Applies the reasoning strategy to solve a problem or answer a query.
    /// </summary>
    /// <param name="query">The problem or question to reason about.</param>
    /// <param name="config">Configuration options for the reasoning process (optional).</param>
    /// <param name="cancellationToken">Token to cancel the operation (optional).</param>
    /// <returns>A reasoning result containing the answer, reasoning chain, and metadata.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is where the actual "thinking" happens. You provide a question
    /// or problem, and the strategy applies its specific reasoning approach to solve it.
    ///
    /// The config parameter lets you customize how the strategy works (like setting how many steps to take,
    /// or how deep to explore). The cancellationToken allows you to stop the reasoning if it's taking too long.
    ///
    /// The result includes not just the final answer, but also the complete reasoning process, so you can
    /// see how the AI arrived at its conclusion.
    /// </para>
    /// </remarks>
    Task<ReasoningResult<T>> ReasonAsync(
        string query,
        ReasoningConfig? config = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the name of this reasoning strategy.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a human-readable name that identifies the strategy,
    /// like "Chain-of-Thought" or "Tree-of-Thoughts". It's useful for logging, debugging,
    /// or displaying to users which reasoning approach was used.
    /// </para>
    /// </remarks>
    string StrategyName { get; }

    /// <summary>
    /// Gets a description of what this reasoning strategy does and when to use it.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This provides information about what makes this strategy unique
    /// and what types of problems it's best suited for. For example, Chain-of-Thought might describe
    /// itself as "Best for problems requiring step-by-step logical deduction."
    /// </para>
    /// </remarks>
    string Description { get; }
}
