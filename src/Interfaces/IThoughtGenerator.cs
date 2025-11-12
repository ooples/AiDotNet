using AiDotNet.Reasoning.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for generating alternative thoughts or reasoning steps.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A thought generator creates new ideas or reasoning steps.
/// Think of it like brainstorming - given where you are now, what are the possible next steps?
///
/// For example, if solving "How to reduce carbon emissions?", the generator might produce:
/// - Thought 1: "Increase renewable energy adoption"
/// - Thought 2: "Improve transportation efficiency"
/// - Thought 3: "Enhance industrial processes"
///
/// Used heavily in Tree-of-Thoughts and other exploratory reasoning strategies.
/// </para>
/// </remarks>
public interface IThoughtGenerator<T>
{
    /// <summary>
    /// Generates alternative thoughts or next steps from the current state.
    /// </summary>
    /// <param name="currentNode">The current thought node to expand from.</param>
    /// <param name="numThoughts">Number of alternative thoughts to generate.</param>
    /// <param name="config">Reasoning configuration.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of new thought nodes.</returns>
    Task<List<ThoughtNode<T>>> GenerateThoughtsAsync(
        ThoughtNode<T> currentNode,
        int numThoughts,
        ReasoningConfig config,
        CancellationToken cancellationToken = default);
}
