using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;

namespace AiDotNet.Reasoning.Search;

/// <summary>
/// Implements Beam Search for exploring reasoning trees with limited memory.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Beam Search is like BFS but only keeps the N most promising paths
/// at each level, where N is the "beam width". This makes it much more memory efficient.
///
/// **Analogy:**
/// Imagine exploring different routes on a map:
/// - BFS: Keep track of EVERY possible route
/// - Beam Search: Only keep track of the 5 most promising routes
///
/// **Characteristics:**
/// - **Memory efficient**: Only stores beam_width nodes per level
/// - **Not guaranteed complete**: Might miss the optimal path
/// - **Good trade-off**: Better than BFS for memory, better than greedy for quality
///
/// **Example with beam_width=2:**
/// ```
/// Problem (depth 0)
///   ├─ Thought A (score: 0.9) ← Top 2 at depth 1
///   ├─ Thought B (score: 0.85) ← Top 2 at depth 1
///   └─ Thought C (score: 0.6) ← Pruned (not in top 2)
///       From A and B, expand top 2 at depth 2...
/// ```
///
/// **When to use:**
/// - Large search spaces where BFS is too slow
/// - When memory is limited
/// - When a good-enough solution is acceptable
/// - Translation, text generation (commonly used here)
///
/// **Research basis:**
/// Used in state-of-the-art models for:
/// - Machine translation
/// - Text generation
/// - Planning systems
/// - Reasoning in large language models
/// </para>
/// </remarks>
public class BeamSearch<T> : ISearchAlgorithm<T>
{
    /// <inheritdoc/>
    public string AlgorithmName => "Beam Search";

    /// <inheritdoc/>
    public string Description =>
        "Maintains a fixed-size beam of most promising nodes at each depth level. " +
        "Memory efficient alternative to BFS with configurable quality/speed tradeoff via beam width.";

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

        if (config.BeamWidth < 1)
            throw new ArgumentException("BeamWidth must be at least 1", nameof(config));

        // Evaluate root
        root.EvaluationScore = await evaluator.EvaluateThoughtAsync(root, root.Thought, config, cancellationToken);
        root.IsVisited = true;

        // Current beam (starts with just the root)
        var currentBeam = new List<AiDotNet.Reasoning.Models.ThoughtNode<T>> { root };
        AiDotNet.Reasoning.Models.ThoughtNode<T>? bestTerminalNode = null;
        var numOps = MathHelper.GetNumericOperations<T>();
        T bestScore = numOps.Zero;

        int nodesExplored = 0;

        // Explore depth by depth
        for (int depth = 0; depth < config.ExplorationDepth; depth++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var nextBeam = new List<AiDotNet.Reasoning.Models.ThoughtNode<T>>();

            // Expand each node in the current beam
            foreach (var node in currentBeam)
            {
                // Check if terminal
                if (node.CheckIsTerminalByHeuristic())
                {
                    node.IsTerminal = true;

                    double nodeScore = Convert.ToDouble(node.EvaluationScore);
                    double bestScoreValue = Convert.ToDouble(bestScore);

                    if (bestTerminalNode == null || nodeScore > bestScoreValue)
                    {
                        bestTerminalNode = node;
                        bestScore = node.EvaluationScore;
                    }

                    continue;
                }

                // Generate children
                var children = await generator.GenerateThoughtsAsync(
                    node,
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

                    node.Children.Add(child);
                    nextBeam.Add(child);
                    nodesExplored++;
                }
            }

            // If no new nodes, we're done
            if (nextBeam.Count == 0)
                break;

            // Keep only top beam_width nodes for next iteration
            currentBeam = nextBeam
                .OrderByDescending(n => Convert.ToDouble(n.EvaluationScore))
                .Take(config.BeamWidth)
                .ToList();

            // Check if we've found good terminal nodes in the beam
            foreach (var node in currentBeam)
            {
                if (node.CheckIsTerminalByHeuristic())
                {
                    node.IsTerminal = true;

                    double nodeScore = Convert.ToDouble(node.EvaluationScore);
                    double bestScoreValue = Convert.ToDouble(bestScore);

                    if (bestTerminalNode == null || nodeScore > bestScoreValue)
                    {
                        bestTerminalNode = node;
                        bestScore = node.EvaluationScore;
                    }
                }
            }
        }

        // Return the best path found
        if (bestTerminalNode != null)
        {
            return ReconstructPath(bestTerminalNode);
        }

        // Fallback: return the highest-scoring path in the final beam
        if (currentBeam.Count > 0)
        {
            var bestNode = currentBeam.OrderByDescending(n => Convert.ToDouble(n.EvaluationScore)).First();
            return ReconstructPath(bestNode);
        }

        // Last resort: return just the root
        return new List<AiDotNet.Reasoning.Models.ThoughtNode<T>> { root };
    }

    /// <summary>
    /// Reconstructs the path from root to a given node.
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
