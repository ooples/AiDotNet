using AiDotNet.Reasoning.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for search algorithms that explore reasoning trees.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A search algorithm determines HOW to explore a tree of possible solutions.
/// Think of it like choosing a strategy for exploring a maze:
///
/// - **Breadth-First Search (BFS)**: Explore all paths at the same depth before going deeper
///   (like exploring all nearby rooms before going down hallways)
///
/// - **Depth-First Search (DFS)**: Follow one path as deep as possible before backtracking
///   (like following one hallway to its end before trying others)
///
/// - **Best-First Search**: Always explore the most promising path next
///   (like always choosing the hallway that looks brightest)
///
/// - **Beam Search**: Keep track of the N most promising paths simultaneously
///   (like exploring your top 5 favorite hallways in parallel)
///
/// - **Monte Carlo Tree Search (MCTS)**: Use random sampling and statistics to guide exploration
///   (like the approach used in game-playing AIs like AlphaGo)
///
/// Different algorithms have different tradeoffs in terms of completeness, optimality, and efficiency.
/// </para>
/// </remarks>
public interface ISearchAlgorithm<T>
{
    /// <summary>
    /// Searches through a thought tree to find the best reasoning path.
    /// </summary>
    /// <param name="root">The root node of the thought tree.</param>
    /// <param name="generator">Generator for creating new thoughts.</param>
    /// <param name="evaluator">Evaluator for scoring thoughts.</param>
    /// <param name="config">Reasoning configuration.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The best path found (list of nodes from root to solution).</returns>
    Task<List<AiDotNet.Reasoning.Models.ThoughtNode<T>>> SearchAsync(
        AiDotNet.Reasoning.Models.ThoughtNode<T> root,
        IThoughtGenerator<T> generator,
        IThoughtEvaluator<T> evaluator,
        ReasoningConfig config,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the name of this search algorithm.
    /// </summary>
    string AlgorithmName { get; }

    /// <summary>
    /// Gets a description of how this algorithm works.
    /// </summary>
    string Description { get; }
}
