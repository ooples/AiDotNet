using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;

namespace AiDotNet.Reasoning.Search;

/// <summary>
/// Implements Breadth-First Search for exploring reasoning trees.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Breadth-First Search (BFS) explores all thoughts at one depth level
/// before moving to the next level. Think of it like exploring a building:
/// - First, explore all rooms on floor 1
/// - Then explore all rooms on floor 2
/// - Then floor 3, and so on
///
/// **Characteristics:**
/// - **Complete**: Will find a solution if one exists
/// - **Optimal**: Finds the shortest path (fewest steps)
/// - **Memory intensive**: Must remember all nodes at current level
///
/// **When to use:**
/// - When you want the shortest reasoning path
/// - When depth is limited
/// - When thoroughness matters more than speed
///
/// **Example flow:**
/// ```
/// Problem (depth 0)
///   ├─ Thought A (depth 1)
///   ├─ Thought B (depth 1)  ← Explore all depth 1 first
///   └─ Thought C (depth 1)
///       ├─ Thought B1 (depth 2)
///       ├─ Thought B2 (depth 2)  ← Then explore depth 2
///       └─ Thought B3 (depth 2)
/// ```
/// </para>
/// </remarks>
public class BreadthFirstSearch<T> : ISearchAlgorithm<T>
{
    /// <inheritdoc/>
    public string AlgorithmName => "Breadth-First Search";

    /// <inheritdoc/>
    public string Description =>
        "Explores all nodes at the current depth before moving deeper. " +
        "Guarantees finding the shortest path but may be memory intensive.";

    /// <inheritdoc/>
    public async Task<List<ThoughtNode<T>>> SearchAsync(
        ThoughtNode<T> root,
        IThoughtGenerator<T> generator,
        IThoughtEvaluator<T> evaluator,
        ReasoningConfig config,
        CancellationToken cancellationToken = default)
    {
        if (root == null)
            throw new ArgumentNullException(nameof(root));

        // Track the best path found
        ThoughtNode<T>? bestTerminalNode = null;
        var numOps = MathHelper.GetNumericOperations<T>();
        T bestScore = numOps.Zero;

        // BFS queue
        var queue = new Queue<ThoughtNode<T>>();
        queue.Enqueue(root);

        // Evaluate root
        root.EvaluationScore = await evaluator.EvaluateThoughtAsync(root, root.Thought, config, cancellationToken);
        root.IsVisited = true;

        int nodesExplored = 0;

        while (queue.Count > 0 && nodesExplored < config.MaxSteps * config.BranchingFactor)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var currentNode = queue.Dequeue();
            nodesExplored++;

            // Check if this is a terminal node
            if (IsTerminalNode(currentNode) || currentNode.Depth >= config.ExplorationDepth)
            {
                currentNode.IsTerminal = true;

                // Update best if this is better
                double currentScoreValue = Convert.ToDouble(currentNode.EvaluationScore);
                double bestScoreValue = Convert.ToDouble(bestScore);

                if (bestTerminalNode == null || currentScoreValue > bestScoreValue)
                {
                    bestTerminalNode = currentNode;
                    bestScore = currentNode.EvaluationScore;
                }

                continue; // Don't expand terminal nodes
            }

            // Generate child thoughts
            var children = await generator.GenerateThoughtsAsync(
                currentNode,
                config.BranchingFactor,
                config,
                cancellationToken);

            // Evaluate each child
            foreach (var child in children)
            {
                child.EvaluationScore = await evaluator.EvaluateThoughtAsync(
                    child,
                    root.Thought,
                    config,
                    cancellationToken);

                currentNode.Children.Add(child);
                queue.Enqueue(child);
            }
        }

        // Return the best path found
        if (bestTerminalNode != null)
        {
            return bestTerminalNode.GetPathFromRoot()
                .Select(thought => FindNodeWithThought(root, thought))
                .Where(n => n != null)
                .Cast<ThoughtNode<T>>()
                .ToList();
        }

        // Fallback: return just the root if no terminal node found
        return new List<ThoughtNode<T>> { root };
    }

    /// <summary>
    /// Checks if a node represents a terminal/complete solution.
    /// </summary>
    private bool IsTerminalNode(ThoughtNode<T> node)
    {
        // Simple heuristic: check for solution indicators in the thought
        string thought = node.Thought.ToLowerInvariant();
        return thought.Contains("final answer") ||
               thought.Contains("conclusion") ||
               thought.Contains("therefore") ||
               thought.Contains("the answer is") ||
               node.IsTerminal;
    }

    /// <summary>
    /// Finds a node in the tree with the given thought text (DFS helper).
    /// </summary>
    private ThoughtNode<T>? FindNodeWithThought(ThoughtNode<T> root, string thought)
    {
        if (root.Thought == thought)
            return root;

        foreach (var child in root.Children)
        {
            var found = FindNodeWithThought(child, thought);
            if (found != null)
                return found;
        }

        return null;
    }
}
