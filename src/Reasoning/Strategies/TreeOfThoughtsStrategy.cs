using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Components;
using AiDotNet.Reasoning.Models;
using AiDotNet.Reasoning.Search;

namespace AiDotNet.Reasoning.Strategies;

/// <summary>
/// Implements Tree-of-Thoughts (ToT) reasoning that explores multiple reasoning paths in a tree structure.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Tree-of-Thoughts (ToT) is like exploring a maze where you can try
/// multiple paths and backtrack if you hit a dead end. Unlike Chain-of-Thought which follows one
/// linear path, ToT explores a tree of possibilities.
///
/// **How it works:**
/// ```
/// Problem: "How can we reduce carbon emissions?"
///
///   ├─ Renewable Energy (score: 0.9)
///   │  ├─ Solar panels on buildings (score: 0.85)
///   │  ├─ Wind farm expansion (score: 0.80)
///   │  └─ Hydroelectric upgrades (score: 0.75)
///   │
///   ├─ Transportation (score: 0.85)
///   │  ├─ Electric vehicles (score: 0.90) ← Best path
///   │  └─ Public transit (score: 0.82)
///   │
///   └─ Industrial (score: 0.75)
///      ├─ Carbon capture (score: 0.70)
///      └─ Process efficiency (score: 0.65)
/// ```
///
/// The search algorithm explores this tree to find the best reasoning path.
///
/// **Key features:**
/// - **Explores multiple paths**: Not limited to one direction
/// - **Can backtrack**: If a path looks bad, try another
/// - **Evaluation at each step**: Score thoughts as you go
/// - **Configurable search**: BFS, Beam Search, or other algorithms
///
/// **Compared to other strategies:**
/// - **Chain-of-Thought**: Linear, one path only
/// - **Self-Consistency**: Multiple independent paths, no tree structure
/// - **Tree-of-Thoughts**: Structured exploration with evaluation and backtracking
///
/// **Research basis:**
/// "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al., 2023)
/// showed significant improvements on planning, math, and creative tasks.
///
/// **When to use:**
/// - Complex problems with multiple viable approaches
/// - When you need to explore and compare alternatives
/// - Planning tasks with branching possibilities
/// - Creative problem-solving
/// - Strategic decision-making
/// </para>
/// <para><b>Example Usage:</b>
/// <code>
/// var chatModel = new OpenAIChatModel&lt;double&gt;("gpt-4");
/// var strategy = new TreeOfThoughtsStrategy&lt;double&gt;(
///     chatModel,
///     searchAlgorithm: SearchAlgorithmType.BeamSearch
/// );
///
/// var config = new ReasoningConfig
/// {
///     ExplorationDepth = 3,      // Explore 3 levels deep
///     BranchingFactor = 3,       // Generate 3 alternatives at each node
///     BeamWidth = 5              // Keep top 5 paths (for Beam Search)
/// };
///
/// var result = await strategy.ReasonAsync(
///     "Design a sustainable city transportation system",
///     config
/// );
///
/// // Result includes the best reasoning path found
/// Console.WriteLine(result.FinalAnswer);
/// Console.WriteLine($"Explored {result.Metrics["nodes_explored"]} possibilities");
/// </code>
/// </para>
/// </remarks>
public class TreeOfThoughtsStrategy<T> : ReasoningStrategyBase<T>
{
    private readonly IThoughtGenerator<T> _generator;
    private readonly IThoughtEvaluator<T> _evaluator;
    private readonly ISearchAlgorithm<T> _searchAlgorithm;

    /// <summary>
    /// Initializes a new instance of the <see cref="TreeOfThoughtsStrategy{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model used for reasoning.</param>
    /// <param name="tools">Optional tools available during reasoning.</param>
    /// <param name="searchAlgorithmType">The search algorithm to use (default: BeamSearch).</param>
    /// <param name="generator">Optional custom thought generator.</param>
    /// <param name="evaluator">Optional custom thought evaluator.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a Tree-of-Thoughts strategy. You can customize:
    /// - The search algorithm (how to explore the tree)
    /// - The thought generator (how to create alternatives)
    /// - The thought evaluator (how to score thoughts)
    ///
    /// If you don't specify custom components, sensible defaults are used.
    /// </para>
    /// </remarks>
    public TreeOfThoughtsStrategy(
        IChatModel<T> chatModel,
        IEnumerable<ITool>? tools = null,
        SearchAlgorithmType searchAlgorithmType = SearchAlgorithmType.BeamSearch,
        IThoughtGenerator<T>? generator = null,
        IThoughtEvaluator<T>? evaluator = null)
        : base(chatModel, tools)
    {
        _generator = generator ?? new ThoughtGenerator<T>(chatModel);
        _evaluator = evaluator ?? new ThoughtEvaluator<T>(chatModel);
        _searchAlgorithm = CreateSearchAlgorithm(searchAlgorithmType);
    }

    /// <inheritdoc/>
    public override string StrategyName => "Tree-of-Thoughts";

    /// <inheritdoc/>
    public override string Description =>
        "Explores multiple reasoning paths in a tree structure using search algorithms. " +
        "Evaluates and compares alternatives to find the best solution path. " +
        "Based on 'Tree of Thoughts' (Yao et al., 2023). " +
        "Best for complex problems requiring exploration and planning.";

    /// <inheritdoc/>
    protected override async Task<ReasoningResult<T>> ReasonCoreAsync(
        string query,
        ReasoningConfig config,
        CancellationToken cancellationToken)
    {
        ValidateConfig(config);

        var result = new ReasoningResult<T>
        {
            StrategyUsed = StrategyName
        };

        AppendTrace($"Starting Tree-of-Thoughts reasoning for query: {query}");
        AppendTrace($"Search algorithm: {_searchAlgorithm.AlgorithmName}");
        AppendTrace($"Max depth: {config.ExplorationDepth}, Branching factor: {config.BranchingFactor}");

        var startTime = DateTime.UtcNow;

        // Step 1: Create root node with the query
        var rootNode = new ThoughtNode<T>
        {
            Thought = query,
            Depth = 0,
            IsVisited = false,
            Parent = null
        };

        AppendTrace("Created root node, starting tree exploration...");

        // Step 2: Execute tree search
        List<ThoughtNode<T>> bestPath;
        try
        {
            bestPath = await _searchAlgorithm.SearchAsync(
                rootNode,
                _generator,
                _evaluator,
                config,
                cancellationToken
            );

            AppendTrace($"Search complete. Best path has {bestPath.Count} nodes.");
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.ErrorMessage = $"Tree search failed: {ex.Message}";
            AppendTrace($"ERROR: {result.ErrorMessage}");
            return result;
        }

        if (bestPath.Count == 0)
        {
            result.Success = false;
            result.ErrorMessage = "No valid reasoning path found";
            AppendTrace($"ERROR: {result.ErrorMessage}");
            return result;
        }

        // Step 3: Convert the best path to a reasoning chain
        var chain = new ReasoningChain<T>
        {
            Query = query,
            StartedAt = startTime
        };

        for (int i = 0; i < bestPath.Count; i++)
        {
            var node = bestPath[i];
            var step = new ReasoningStep<T>
            {
                StepNumber = i + 1,
                Content = node.Thought,
                Score = node.EvaluationScore,
                IsVerified = true, // Thoughts were evaluated during search
                CreatedAt = DateTime.UtcNow
            };

            if (node.Metadata.ContainsKey("evaluation_response"))
            {
                step.Metadata["evaluation"] = node.Metadata["evaluation_response"];
            }

            chain.AddStep(step);
        }

        // Step 4: Extract or generate final answer from the best path
        string finalAnswer = await ExtractFinalAnswerAsync(bestPath, query, config, cancellationToken);
        chain.FinalAnswer = finalAnswer;
        chain.CompletedAt = DateTime.UtcNow;

        // Step 5: Calculate overall score (average of path scores)
        if (chain.Steps.Count > 0)
        {
            chain.OverallScore = chain.GetAverageScore();
        }

        // Step 6: Collect metrics about the tree exploration
        int totalNodes = CountNodesInTree(rootNode);
        int maxDepthReached = bestPath.Count > 0 ? bestPath.Max(n => n.Depth) : 0;

        AppendTrace($"\nTree exploration statistics:");
        AppendTrace($"  Total nodes explored: {totalNodes}");
        AppendTrace($"  Max depth reached: {maxDepthReached}");
        AppendTrace($"  Best path length: {bestPath.Count}");
        AppendTrace($"  Average score in best path: {chain.OverallScore}");

        // Step 7: Build final result
        result.FinalAnswer = finalAnswer;
        result.ReasoningChain = chain;
        result.OverallConfidence = chain.OverallScore;
        result.Success = !string.IsNullOrWhiteSpace(finalAnswer);
        result.Metrics["nodes_explored"] = totalNodes;
        result.Metrics["max_depth_reached"] = maxDepthReached;
        result.Metrics["best_path_length"] = bestPath.Count;
        result.Metrics["branching_factor"] = config.BranchingFactor;
        result.Metrics["search_algorithm"] = _searchAlgorithm.AlgorithmName;

        AppendTrace($"\nTree-of-Thoughts reasoning complete: {finalAnswer}");

        return result;
    }

    /// <summary>
    /// Extracts or generates the final answer from the best reasoning path.
    /// </summary>
    private async Task<string> ExtractFinalAnswerAsync(
        List<ThoughtNode<T>> path,
        string originalQuery,
        ReasoningConfig config,
        CancellationToken cancellationToken)
    {
        if (path.Count == 0)
            return "No solution found";

        // If the last node is marked as terminal, use its thought
        var lastNode = path[^1];
        if (lastNode.IsTerminal)
        {
            return lastNode.Thought;
        }

        // Otherwise, ask the LLM to synthesize a final answer from the path
        string pathSummary = string.Join(" → ", path.Select(n => n.Thought));

        string prompt = $@"Based on the following reasoning path, provide a clear final answer to the original question.

Original Question: {originalQuery}

Reasoning Path:
{pathSummary}

Provide a concise final answer:";

        string answer = await ChatModel.GenerateResponseAsync(prompt);
        return answer.Trim();
    }

    /// <summary>
    /// Counts total nodes in the tree (DFS).
    /// </summary>
    private int CountNodesInTree(ThoughtNode<T> root)
    {
        int count = 1; // Count this node

        foreach (var child in root.Children)
        {
            count += CountNodesInTree(child);
        }

        return count;
    }

    /// <summary>
    /// Creates the appropriate search algorithm based on type.
    /// </summary>
    private ISearchAlgorithm<T> CreateSearchAlgorithm(SearchAlgorithmType type)
    {
        return type switch
        {
            SearchAlgorithmType.BreadthFirst => new BreadthFirstSearch<T>(),
            SearchAlgorithmType.BeamSearch => new BeamSearch<T>(),
            _ => new BeamSearch<T>() // Default to BeamSearch
        };
    }
}

/// <summary>
/// Types of search algorithms available for Tree-of-Thoughts.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Different ways to explore the tree of possibilities.
/// Each has different tradeoffs in terms of speed, memory, and quality.
/// </para>
/// </remarks>
public enum SearchAlgorithmType
{
    /// <summary>
    /// Breadth-First Search - explores all nodes at each level before going deeper.
    /// Thorough but memory intensive.
    /// </summary>
    BreadthFirst,

    /// <summary>
    /// Beam Search - keeps only the top N most promising paths at each level.
    /// Good balance of quality and efficiency (recommended default).
    /// </summary>
    BeamSearch
}
