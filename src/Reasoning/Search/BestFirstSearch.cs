using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;

namespace AiDotNet.Reasoning.Search;

/// <summary>
/// Implements Best-First Search (greedy) for reasoning tree exploration.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Best-First Search is a greedy algorithm that always explores
/// the most promising node next, based on evaluation scores. Think of it like always taking
/// the path that looks best right now.
///
/// **How it works:**
/// 1. Keep a priority queue of all discovered nodes
/// 2. Always expand the node with the highest score
/// 3. Add its children to the queue
/// 4. Repeat until terminal or depth limit
///
/// **Characteristics:**
/// - **Fast**: Makes quick decisions based on current information
/// - **Greedy**: Might get stuck in local optima
/// - **Good heuristic needed**: Quality depends on evaluation function
///
/// **Comparison:**
/// - BFS: Explores by level (ignores quality)
/// - DFS: Explores deeply (ignores quality)
/// - Best-First: Always picks highest-scored node (ignores structure)
/// - Beam Search: Best-First but limits to top K nodes
///
/// **When to use:**
/// - When you have a good evaluation function
/// - When you want fast solutions
/// - When the best path is likely to look good early
/// - Planning and heuristic search problems
///
/// **Example:**
/// ```
/// Queue: [A(0.9), B(0.7), C(0.6)]
/// Pick A (highest score)
/// Expand A → children [A1(0.95), A2(0.8)]
/// Queue: [A1(0.95), A2(0.8), B(0.7), C(0.6)]
/// Pick A1 (highest score)
/// Continue...
/// ```
/// </para>
/// </remarks>
public class BestFirstSearch<T> : ISearchAlgorithm<T>
{
    /// <inheritdoc/>
    public string AlgorithmName => "Best-First Search";

    /// <inheritdoc/>
    public string Description =>
        "Greedy algorithm that always expands the highest-scored node. " +
        "Fast but may miss optimal solutions. Quality depends on evaluation function.";

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

        // Priority queue with unique counter to prevent duplicate dropping
        int nodeCounter = 0;
        var priorityQueue = new SortedSet<(AiDotNet.Reasoning.Models.ThoughtNode<T> node, int id)>(
            Comparer<(AiDotNet.Reasoning.Models.ThoughtNode<T> node, int id)>.Create((a, b) =>
            {
                // Primary: score (descending)
                double scoreA = Convert.ToDouble(a.node.EvaluationScore);
                double scoreB = Convert.ToDouble(b.node.EvaluationScore);
                int scoreCompare = scoreB.CompareTo(scoreA);
                if (scoreCompare != 0) return scoreCompare;

                // Secondary: depth (ascending - prefer shallower)
                int depthCompare = a.node.Depth.CompareTo(b.node.Depth);
                if (depthCompare != 0) return depthCompare;

                // Tertiary: insertion order (guaranteed unique)
                return a.id.CompareTo(b.id);
            })
        );

        priorityQueue.Add((root, nodeCounter++));

        AiDotNet.Reasoning.Models.ThoughtNode<T>? bestTerminalNode = null;
        var numOps = MathHelper.GetNumericOperations<T>();
        T bestScore = numOps.Zero;
        int nodesExpanded = 0;

        while (priorityQueue.Count > 0 && nodesExpanded < config.MaxSteps * config.BranchingFactor)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Pick best node (extract from tuple)
            var (currentNode, _) = priorityQueue.First();
            priorityQueue.Remove(priorityQueue.First());
            nodesExpanded++;

            // Check if terminal
            if (currentNode.CheckIsTerminalByHeuristic() || currentNode.Depth >= config.ExplorationDepth)
            {
                currentNode.IsTerminal = true;

                double nodeScore = Convert.ToDouble(currentNode.EvaluationScore);
                double currentBest = Convert.ToDouble(bestScore);

                if (bestTerminalNode == null || nodeScore > currentBest)
                {
                    bestTerminalNode = currentNode;
                    bestScore = currentNode.EvaluationScore;
                }

                continue;
            }

            // Expand: generate children
            var children = await generator.GenerateThoughtsAsync(
                currentNode,
                config.BranchingFactor,
                config,
                cancellationToken
            );

            // Evaluate and add children to queue (with unique counter)
            foreach (var child in children)
            {
                child.EvaluationScore = await evaluator.EvaluateThoughtAsync(
                    child,
                    child.Thought,
                    config,
                    cancellationToken
                );

                currentNode.Children.Add(child);
                priorityQueue.Add((child, nodeCounter++));
            }
        }

        // Return best path found
        if (bestTerminalNode != null)
        {
            return ReconstructPath(bestTerminalNode);
        }

        // Fallback: return highest-scored node in queue (extract from tuple)
        if (priorityQueue.Count > 0)
        {
            var (bestNode, _) = priorityQueue.First();
            return ReconstructPath(bestNode);
        }

        return new List<AiDotNet.Reasoning.Models.ThoughtNode<T>> { root };
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
