using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Reasoning.Search;

/// <summary>
/// Implements Monte Carlo Tree Search (MCTS) for reasoning tree exploration.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Monte Carlo Tree Search (MCTS) is the algorithm that powered AlphaGo
/// to beat the world champion at Go. It balances exploration (trying new paths) with exploitation
/// (following promising paths).
///
/// **How it works:**
/// 1. **Selection**: Start at root, pick most promising child using UCB1 formula
/// 2. **Expansion**: Add new child nodes to expand the tree
/// 3. **Simulation**: Run quick "rollout" to estimate value
/// 4. **Backpropagation**: Update scores back up the tree
/// 5. Repeat for many iterations
///
/// **Key formula (UCB1):**
/// Score = exploitation + exploration
/// Score = avg_value + C * sqrt(ln(parent_visits) / node_visits)
///
/// **When to use:**
/// - Complex problems with large search spaces
/// - When you need good (not necessarily optimal) solutions
/// - When you have computational budget for many iterations
/// - Game playing, planning, strategic reasoning
///
/// **Famous uses:**
/// - AlphaGo / AlphaZero (game playing)
/// - Planning algorithms
/// - Decision making under uncertainty
///
/// **Research:**
/// "A Survey of Monte Carlo Tree Search Methods" (Browne et al., 2012)
/// </para>
/// </remarks>
internal class MonteCarloTreeSearch<T> : ISearchAlgorithm<T>
{
    private readonly double _explorationConstant;
    private readonly int _numSimulations;
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the <see cref="MonteCarloTreeSearch{T}"/> class.
    /// </summary>
    /// <param name="explorationConstant">UCB1 exploration constant (default: 1.414, sqrt(2)). Must be non-negative.</param>
    /// <param name="numSimulations">Number of MCTS iterations (default: 100). Must be at least 1.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if explorationConstant is negative or numSimulations is less than 1.</exception>
    public MonteCarloTreeSearch(double explorationConstant = 1.414, int numSimulations = 100)
    {
        if (explorationConstant < 0)
            throw new ArgumentOutOfRangeException(nameof(explorationConstant), "Exploration constant must be non-negative.");
        if (numSimulations < 1)
            throw new ArgumentOutOfRangeException(nameof(numSimulations), "Number of simulations must be at least 1.");

        _explorationConstant = explorationConstant;
        _numSimulations = numSimulations;
        _random = RandomHelper.CreateSeededRandom(42); // Deterministic for reproducibility
    }

    /// <inheritdoc/>
    public string AlgorithmName => "Monte Carlo Tree Search (MCTS)";

    /// <inheritdoc/>
    public string Description =>
        $"Balances exploration and exploitation using UCB1. Runs {_numSimulations} simulations. " +
        "Used in AlphaGo and strategic planning.";

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

        // Initialize root
        root.EvaluationScore = await evaluator.EvaluateThoughtAsync(root, root.Thought, config, cancellationToken);
        root.Metadata["visits"] = 0;
        root.Metadata["total_value"] = 0.0;

        // Run MCTS iterations
        for (int i = 0; i < _numSimulations; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // 1. Selection: Navigate to leaf using UCB1
            var node = Select(root);

            // 2. Expansion: Add children if not terminal
            if (!node.IsTerminal && node.Depth < config.ExplorationDepth)
            {
                var children = await generator.GenerateThoughtsAsync(node, config.BranchingFactor, config, cancellationToken);

                foreach (var child in children)
                {
                    child.EvaluationScore = await evaluator.EvaluateThoughtAsync(child, child.Thought, config, cancellationToken);
                    child.Metadata["visits"] = 0;
                    child.Metadata["total_value"] = 0.0;
                    node.Children.Add(child);
                }

                // Use the expanded node itself for simulation (standard MCTS approach)
                // This avoids positional bias from always selecting the first child
            }

            // 3. Simulation: Estimate value (use evaluation score)
            double value = Convert.ToDouble(node.EvaluationScore);

            // Check if terminal
            if (node.CheckIsTerminalByHeuristic())
            {
                node.IsTerminal = true;
                value *= 1.5; // Boost terminal nodes
            }

            // 4. Backpropagation: Update statistics up the tree
            Backpropagate(node, value);
        }

        // Select best path based on visit counts (most explored = most promising)
        var bestPath = new List<AiDotNet.Reasoning.Models.ThoughtNode<T>>();
        var current = root;

        while (current != null && current.Children.Count > 0)
        {
            bestPath.Add(current);

            // Pick child with most visits
            current = current.Children
                .OrderByDescending(c => c.Metadata.ContainsKey("visits") ? (int)c.Metadata["visits"] : 0)
                .FirstOrDefault();

            if (current != null && current.IsTerminal)
            {
                bestPath.Add(current);
                break;
            }
        }

        return bestPath.Count > 0 ? bestPath : new List<AiDotNet.Reasoning.Models.ThoughtNode<T>> { root };
    }

    /// <summary>
    /// Selection phase: Navigate to most promising leaf using UCB1.
    /// </summary>
    private AiDotNet.Reasoning.Models.ThoughtNode<T> Select(AiDotNet.Reasoning.Models.ThoughtNode<T> node)
    {
        while (node.Children.Count > 0)
        {
            // Pick child with highest UCB1 score
            node = node.Children
                .OrderByDescending(c => CalculateUCB1(c, node))
                .First();

            // Stop if unvisited or terminal
            if (!node.Metadata.ContainsKey("visits") || (int)node.Metadata["visits"] == 0 || node.IsTerminal)
            {
                break;
            }
        }

        return node;
    }

    /// <summary>
    /// Calculates UCB1 score for a node.
    /// </summary>
    private double CalculateUCB1(AiDotNet.Reasoning.Models.ThoughtNode<T> node, AiDotNet.Reasoning.Models.ThoughtNode<T> parent)
    {
        int visits = node.Metadata.ContainsKey("visits") ? (int)node.Metadata["visits"] : 0;
        int parentVisits = parent.Metadata.ContainsKey("visits") ? (int)parent.Metadata["visits"] : 1;

        if (visits == 0)
        {
            return double.MaxValue; // Prioritize unvisited nodes
        }

        double totalValue = node.Metadata.ContainsKey("total_value") ? (double)node.Metadata["total_value"] : 0.0;
        double avgValue = totalValue / visits;

        // UCB1 formula: exploitation + exploration
        double exploitation = avgValue;
        double exploration = _explorationConstant * Math.Sqrt(Math.Log(parentVisits) / visits);

        return exploitation + exploration;
    }

    /// <summary>
    /// Backpropagation phase: Update node statistics up the tree.
    /// </summary>
    private void Backpropagate(AiDotNet.Reasoning.Models.ThoughtNode<T> node, double value)
    {
        var current = node;

        while (current != null)
        {
            int visits = current.Metadata.ContainsKey("visits") ? (int)current.Metadata["visits"] : 0;
            double totalValue = current.Metadata.ContainsKey("total_value") ? (double)current.Metadata["total_value"] : 0.0;

            current.Metadata["visits"] = visits + 1;
            current.Metadata["total_value"] = totalValue + value;

            current = current.Parent;
        }
    }
}
