using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;

namespace AiDotNet.Reasoning.Search;

/// <summary>
/// Helper class to track the best result during DFS traversal.
/// </summary>
internal class DFSBestResult<T>
{
    public AiDotNet.Reasoning.Models.ThoughtNode<T>? BestTerminal { get; set; }
    public T BestScore { get; set; }

    public DFSBestResult(T initialScore)
    {
        BestScore = initialScore;
    }
}

/// <summary>
/// Implements Depth-First Search for exploring reasoning trees.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Depth-First Search (DFS) explores one path all the way to the end
/// before backtracking to try other paths. Think of it like exploring a maze:
/// - Follow one corridor all the way to its end
/// - If it's a dead end, backtrack and try another corridor
/// - Keep going until you find the exit
///
/// **Characteristics:**
/// - **Memory efficient**: Only stores current path
/// - **Can miss optimal**: Might find a solution but not the best one
/// - **Good for deep problems**: Explores deeply before breadth
///
/// **When to use:**
/// - When solutions are deep in the tree
/// - When memory is limited
/// - When any solution is acceptable (not necessarily optimal)
///
/// **Example:**
/// ```
/// Problem (depth 0)
///   ├─ Thought A (depth 1)
///   │  └─ Thought A1 (depth 2)
///   │     └─ Thought A1a (depth 3) ← Explore all the way down first
///   ├─ Thought B (depth 1)  ← Then backtrack and try this
///   └─ Thought C (depth 1)
/// ```
/// </para>
/// </remarks>
internal class DepthFirstSearch<T> : ISearchAlgorithm<T>
{
    /// <inheritdoc/>
    public string AlgorithmName => "Depth-First Search";

    /// <inheritdoc/>
    public string Description =>
        "Explores paths deeply before backtracking. Memory efficient but may not find optimal solution.";

    /// <inheritdoc/>
    public async Task<List<AiDotNet.Reasoning.Models.ThoughtNode<T>>> SearchAsync(
        AiDotNet.Reasoning.Models.ThoughtNode<T> root,
        IThoughtGenerator<T> generator,
        IThoughtEvaluator<T> evaluator,
        ReasoningConfig config,
        CancellationToken cancellationToken = default)
    {
        if (root == null)
            throw new ArgumentNullException(nameof(root));
        if (generator == null)
            throw new ArgumentNullException(nameof(generator));
        if (evaluator == null)
            throw new ArgumentNullException(nameof(evaluator));
        if (config == null)
            throw new ArgumentNullException(nameof(config));

        // Evaluate root
        root.EvaluationScore = await evaluator.EvaluateThoughtAsync(root, root.Thought, config, cancellationToken);
        root.IsVisited = true;

        // Track best terminal node found
        var numOps = MathHelper.GetNumericOperations<T>();
        var bestResult = new DFSBestResult<T>(numOps.Zero);

        // DFS with recursion
        await DFSRecursive(root, root.Thought, generator, evaluator, config, 0, bestResult, cancellationToken);

        // Return best path
        if (bestResult.BestTerminal != null)
        {
            return ReconstructPath(bestResult.BestTerminal);
        }

        return new List<AiDotNet.Reasoning.Models.ThoughtNode<T>> { root };
    }

    private async Task DFSRecursive(
        AiDotNet.Reasoning.Models.ThoughtNode<T> node,
        string originalQuery,
        IThoughtGenerator<T> generator,
        IThoughtEvaluator<T> evaluator,
        ReasoningConfig config,
        int currentDepth,
        DFSBestResult<T> bestResult,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        // Check if terminal
        if (node.CheckIsTerminalByHeuristic() || currentDepth >= config.ExplorationDepth)
        {
            node.IsTerminal = true;

            // Update best if better
            double nodeScore = Convert.ToDouble(node.EvaluationScore);
            double currentBest = Convert.ToDouble(bestResult.BestScore);

            if (bestResult.BestTerminal == null || nodeScore > currentBest)
            {
                bestResult.BestTerminal = node;
                bestResult.BestScore = node.EvaluationScore;
            }

            return;
        }

        // Generate children
        var children = await generator.GenerateThoughtsAsync(node, config.BranchingFactor, config, cancellationToken);

        // Evaluate and add children
        foreach (var child in children)
        {
            child.EvaluationScore = await evaluator.EvaluateThoughtAsync(child, originalQuery, config, cancellationToken);
            node.Children.Add(child);
        }

        // Recursively explore each child (DFS order)
        foreach (var child in node.Children)
        {
            await DFSRecursive(child, originalQuery, generator, evaluator, config, currentDepth + 1, bestResult, cancellationToken);
        }
    }

    private List<AiDotNet.Reasoning.Models.ThoughtNode<T>> ReconstructPath(AiDotNet.Reasoning.Models.ThoughtNode<T> node)
    {
        var path = new List<AiDotNet.Reasoning.Models.ThoughtNode<T>>();
        var current = node;

        while (current != null)
        {
            path.Insert(0, current);
            current = current.Parent;
        }

        return path;
    }
}
