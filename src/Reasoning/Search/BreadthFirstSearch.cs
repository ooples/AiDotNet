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
internal class BreadthFirstSearch<T> : ISearchAlgorithm<T>
{
    /// <inheritdoc/>
    public string AlgorithmName => "Breadth-First Search";

    /// <inheritdoc/>
    public string Description =>
        "Explores all nodes at the current depth before moving deeper. " +
        "Guarantees finding the shortest path but may be memory intensive.";

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

        // Track the best path found
        AiDotNet.Reasoning.Models.ThoughtNode<T>? bestTerminalNode = null;
        var numOps = MathHelper.GetNumericOperations<T>();
        T bestScore = numOps.Zero;

        // BFS queue
        var queue = new Queue<AiDotNet.Reasoning.Models.ThoughtNode<T>>();
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
            if (currentNode.CheckIsTerminalByHeuristic() || currentNode.Depth >= config.ExplorationDepth)
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

            // Evaluate each child against the original problem
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
            return ReconstructPath(bestTerminalNode);
        }

        // Fallback: find the best explored non-terminal node
        var allNodes = new List<AiDotNet.Reasoning.Models.ThoughtNode<T>>();
        CollectAllNodes(root, allNodes);

        var bestExplored = allNodes
            .Where(n => n.IsVisited)
            .OrderByDescending(n => Convert.ToDouble(n.EvaluationScore))
            .FirstOrDefault();

        if (bestExplored != null && bestExplored != root)
        {
            return ReconstructPath(bestExplored);
        }

        // Last resort: return just the root
        return new List<AiDotNet.Reasoning.Models.ThoughtNode<T>> { root };
    }

    /// <summary>
    /// Collects all nodes in the tree using an iterative approach to avoid StackOverflow on deep trees.
    /// </summary>
    private void CollectAllNodes(AiDotNet.Reasoning.Models.ThoughtNode<T> node, List<AiDotNet.Reasoning.Models.ThoughtNode<T>> collection)
    {
        // Use iterative BFS with explicit queue instead of recursion to prevent StackOverflow
        var nodeQueue = new Queue<AiDotNet.Reasoning.Models.ThoughtNode<T>>();
        nodeQueue.Enqueue(node);

        while (nodeQueue.Count > 0)
        {
            var current = nodeQueue.Dequeue();
            collection.Add(current);

            foreach (var child in current.Children)
            {
                nodeQueue.Enqueue(child);
            }
        }
    }

    /// <summary>
    /// Reconstructs the path from root to a given node by traversing Parent pointers.
    /// </summary>
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
