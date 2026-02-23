using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;
using AiDotNet.Reasoning.Strategies;
using AiDotNet.Validation;

namespace AiDotNet.Reasoning;

/// <summary>
/// The main entry point for AI reasoning capabilities. Provides a simple, unified API for
/// solving problems using advanced reasoning strategies like Chain-of-Thought and Tree-of-Thoughts.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The Reasoner is your one-stop shop for making AI "think step by step."
/// Instead of getting quick, sometimes wrong answers, the Reasoner makes the AI show its work,
/// explore multiple approaches, and verify its reasoning - just like a careful student would.
///
/// **Why use the Reasoner?**
/// - Better accuracy on complex problems (math, logic, coding)
/// - Transparent reasoning you can review and trust
/// - Automatic strategy selection based on problem type
/// - Built-in verification to catch errors
///
/// **Simple Example:**
/// <code>
/// // Create a reasoner with your chat model
/// var reasoner = new Reasoner&lt;double&gt;(chatModel);
///
/// // Solve a math problem with step-by-step reasoning
/// var result = await reasoner.SolveAsync("What is 15% of 240?");
/// Console.WriteLine(result.FinalAnswer);  // "36"
///
/// // See the reasoning steps
/// foreach (var step in result.ReasoningChain.Steps)
/// {
///     Console.WriteLine($"Step {step.StepNumber}: {step.Content}");
/// }
/// </code>
///
/// **Advanced Example:**
/// <code>
/// // Use Tree-of-Thoughts for complex problems
/// var result = await reasoner.SolveAsync(
///     "Design a sustainable transportation system for a city of 1 million people",
///     ReasoningMode.TreeOfThoughts,
///     new ReasoningConfig { ExplorationDepth = 5, EnableVerification = true }
/// );
/// </code>
/// </para>
/// </remarks>
internal class Reasoner<T> : IReasoner<T>
{
    private readonly IChatModel<T> _chatModel;
    private readonly IEnumerable<ITool>? _tools;

    /// <summary>
    /// Initializes a new instance of the <see cref="Reasoner{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model to use for reasoning (e.g., GPT-4, Claude).</param>
    /// <param name="tools">Optional tools available during reasoning (calculators, code execution, etc.).</param>
    /// <exception cref="ArgumentNullException">Thrown when chatModel is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Create a Reasoner by providing your AI chat model.
    /// The chat model is the "brain" that does the thinking - models like GPT-4 or Claude
    /// work best for complex reasoning tasks.
    ///
    /// Optionally, you can provide tools like calculators or code interpreters that
    /// the reasoner can use to verify calculations or run code.
    /// </para>
    /// </remarks>
    public Reasoner(IChatModel<T> chatModel, IEnumerable<ITool>? tools = null)
    {
        Guard.NotNull(chatModel);
        _chatModel = chatModel;
        _tools = tools;
    }

    /// <inheritdoc/>
    public async Task<ReasoningResult<T>> SolveAsync(
        string problem,
        ReasoningMode mode = ReasoningMode.Auto,
        ReasoningConfig? config = null,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(problem))
            throw new ArgumentException("Problem cannot be null or empty", nameof(problem));

        config ??= new ReasoningConfig();

        // Select the appropriate strategy based on mode
        var strategy = CreateStrategy(mode, config);

        // Execute reasoning
        return await strategy.ReasonAsync(problem, config, cancellationToken);
    }

    /// <inheritdoc/>
    public async Task<string> QuickSolveAsync(
        string problem,
        CancellationToken cancellationToken = default)
    {
        // Fast config: minimal steps and no verification
        var fastConfig = new ReasoningConfig
        {
            MaxSteps = 5,
            ExplorationDepth = 1,
            NumSamples = 1,
            BeamWidth = 2,
            EnableVerification = false,
            EnableTestTimeCompute = false,
            MaxReasoningTimeSeconds = 10
        };

        var result = await SolveAsync(
            problem,
            ReasoningMode.ChainOfThought,
            fastConfig,
            cancellationToken);

        return result.FinalAnswer;
    }

    /// <inheritdoc/>
    public async Task<ReasoningResult<T>> DeepSolveAsync(
        string problem,
        CancellationToken cancellationToken = default)
    {
        // Thorough config: extensive exploration and verification
        var thoroughConfig = new ReasoningConfig
        {
            MaxSteps = 20,
            ExplorationDepth = 5,
            BranchingFactor = 5,
            NumSamples = 10,
            BeamWidth = 10,
            Temperature = 0.5,
            EnableVerification = true,
            EnableSelfRefinement = true,
            MaxRefinementAttempts = 3,
            EnableExternalVerification = true,
            EnableTestTimeCompute = true,
            ComputeScalingFactor = 3.0,
            EnableContradictionDetection = true,
            EnableDiversitySampling = true,
            MaxReasoningTimeSeconds = 300
        };

        return await SolveAsync(
            problem,
            ReasoningMode.TreeOfThoughts,
            thoroughConfig,
            cancellationToken);
    }

    /// <inheritdoc/>
    public async Task<ReasoningResult<T>> SolveWithConsensusAsync(
        string problem,
        int numAttempts = 5,
        CancellationToken cancellationToken = default)
    {
        var config = new ReasoningConfig
        {
            NumSamples = numAttempts,
            EnableVerification = true
        };

        return await SolveAsync(
            problem,
            ReasoningMode.SelfConsistency,
            config,
            cancellationToken);
    }

    /// <summary>
    /// Creates the appropriate reasoning strategy based on the selected mode.
    /// </summary>
    private IReasoningStrategy<T> CreateStrategy(ReasoningMode mode, ReasoningConfig config)
    {
        // Auto-select based on config hints
        if (mode == ReasoningMode.Auto)
        {
            mode = SelectOptimalMode(config);
        }

        return mode switch
        {
            ReasoningMode.ChainOfThought => new ChainOfThoughtStrategy<T>(_chatModel, _tools),
            ReasoningMode.TreeOfThoughts => new TreeOfThoughtsStrategy<T>(_chatModel, _tools),
            ReasoningMode.SelfConsistency => new SelfConsistencyStrategy<T>(_chatModel, _tools),
            _ => new ChainOfThoughtStrategy<T>(_chatModel, _tools)
        };
    }

    /// <summary>
    /// Selects the optimal reasoning mode based on configuration.
    /// </summary>
    private ReasoningMode SelectOptimalMode(ReasoningConfig config)
    {
        // If exploration depth is high, use tree-based reasoning
        if (config.ExplorationDepth > 2 || config.BranchingFactor > 2)
        {
            return ReasoningMode.TreeOfThoughts;
        }

        // If multiple samples requested, use self-consistency
        if (config.NumSamples > 1)
        {
            return ReasoningMode.SelfConsistency;
        }

        // Default to Chain-of-Thought for simplicity
        return ReasoningMode.ChainOfThought;
    }
}

/// <summary>
/// Available reasoning modes that determine how problems are solved.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Different problems benefit from different thinking approaches.
/// Just like you might use different study strategies for math vs. essay writing,
/// the AI can use different reasoning modes for different types of problems.
/// </para>
/// </remarks>
public enum ReasoningMode
{
    /// <summary>
    /// Automatically selects the best reasoning mode based on configuration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Let the system choose the best approach.
    /// This is the recommended default for most users.
    /// </para>
    /// </remarks>
    Auto,

    /// <summary>
    /// Linear step-by-step reasoning. Best for straightforward problems.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like showing your work on a math problem.
    /// The AI thinks through the problem one step at a time.
    /// Fast and effective for most problems.
    /// </para>
    /// </remarks>
    ChainOfThought,

    /// <summary>
    /// Explores multiple reasoning paths in a tree structure. Best for complex problems.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like exploring a maze by trying multiple paths.
    /// The AI considers different approaches and picks the best one.
    /// More thorough but uses more compute.
    /// </para>
    /// </remarks>
    TreeOfThoughts,

    /// <summary>
    /// Solves the problem multiple times and uses majority voting. Best for reliability.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like asking 5 people to solve the same problem
    /// and going with the most common answer. Increases confidence but takes longer.
    /// </para>
    /// </remarks>
    SelfConsistency
}
