using System.Diagnostics;
using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;
using AiDotNet.Validation;

namespace AiDotNet.Reasoning;

/// <summary>
/// Abstract base class for reasoning strategies that solve problems through structured thinking.
/// Provides common functionality for all reasoning approaches.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations and scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This base class is like a template for creating different types of reasoning strategies.
/// Just like AgentBase provides common functionality for all agents, ReasoningStrategyBase provides the shared
/// foundation that all reasoning strategies need:
/// - Managing the language model (the "brain")
/// - Tracking tools that can be used
/// - Recording the reasoning process
/// - Handling configuration and timing
///
/// Specific reasoning strategies (Chain-of-Thought, Tree-of-Thoughts, etc.) inherit from this class
/// and implement their unique reasoning logic while getting all the common features for free.
///
/// This follows the Template Method design pattern, where the base class defines the structure
/// and derived classes fill in the specific details.
/// </para>
/// <para><b>Example Usage:</b>
/// <code>
/// // Inherit from this base class to create a custom strategy
/// public class MyCustomStrategy&lt;T&gt; : ReasoningStrategyBase&lt;T&gt;
/// {
///     public MyCustomStrategy(IChatModel&lt;T&gt; chatModel) : base(chatModel) { }
///
///     protected override async Task&lt;ReasoningResult&lt;T&gt;&gt; ReasonCoreAsync(
///         string query, ReasoningConfig config, CancellationToken cancellationToken)
///     {
///         // Implement your custom reasoning logic here
///         AppendTrace("Starting my custom reasoning approach...");
///         // ... your implementation ...
///         return result;
///     }
/// }
/// </code>
/// </para>
/// </remarks>
public abstract class ReasoningStrategyBase<T> : IReasoningStrategy<T>
{
    private readonly System.Text.StringBuilder _reasoningTrace;
    private readonly List<ITool> _tools;
    private readonly object _traceLock = new object();

    /// <summary>
    /// Initializes a new instance of the <see cref="ReasoningStrategyBase{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model used for reasoning and generating thoughts.</param>
    /// <param name="tools">Optional tools that can be used during reasoning (calculators, search, etc.).</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="chatModel"/> is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the basic infrastructure every reasoning
    /// strategy needs:
    /// - A language model (the AI that does the thinking)
    /// - Optional tools (like calculators, code interpreters, search engines)
    /// - A trace/log to record the reasoning process
    ///
    /// Think of it like setting up a workstation: you need a brain (the model), some tools
    /// (calculator, reference books), and a notebook (the trace) to write down your thinking.
    /// </para>
    /// </remarks>
    protected ReasoningStrategyBase(IChatModel<T> chatModel, IEnumerable<ITool>? tools = null)
    {
        Guard.NotNull(chatModel);
        ChatModel = chatModel;
        _tools = tools?.ToList() ?? new List<ITool>();
        _reasoningTrace = new System.Text.StringBuilder();
    }

    /// <summary>
    /// Gets the chat model used for reasoning.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the AI language model that generates thoughts and reasoning steps.
    /// It's protected so derived classes can use it, but not publicly accessible from outside.
    /// </para>
    /// </remarks>
    protected IChatModel<T> ChatModel { get; }

    /// <summary>
    /// Gets the read-only list of tools available to this strategy.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are external tools (like calculators, search engines, code interpreters)
    /// that the reasoning strategy can use to help solve problems. Protected so derived classes can access them.
    /// </para>
    /// </remarks>
    protected IReadOnlyList<ITool> Tools => _tools.AsReadOnly();

    /// <summary>
    /// Gets the current reasoning trace.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a running log of the reasoning process - like a notebook
    /// where all thoughts, actions, and observations are recorded. Useful for debugging and understanding
    /// how the strategy arrived at its answer.
    /// </para>
    /// </remarks>
    protected string ReasoningTrace => _reasoningTrace.ToString();

    /// <inheritdoc/>
    public abstract string StrategyName { get; }

    /// <inheritdoc/>
    public abstract string Description { get; }

    /// <inheritdoc/>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main entry point for reasoning. It:
    /// 1. Sets up timing and configuration
    /// 2. Calls the core reasoning logic (implemented by derived classes)
    /// 3. Handles errors and timeouts gracefully
    /// 4. Records performance metrics
    ///
    /// You call this method to solve a problem, and it returns a comprehensive result.
    /// The actual reasoning strategy is implemented in ReasonCoreAsync by derived classes.
    /// </para>
    /// </remarks>
    public async Task<ReasoningResult<T>> ReasonAsync(
        string query,
        ReasoningConfig? config = null,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or whitespace", nameof(query));

        config ??= new ReasoningConfig();
        ClearTrace();

        var stopwatch = Stopwatch.StartNew();
        var result = new ReasoningResult<T>
        {
            StrategyUsed = StrategyName
        };

        try
        {
            // Apply timeout if configured
            if (config.MaxReasoningTimeSeconds > 0)
            {
                using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(config.MaxReasoningTimeSeconds));
                using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, timeoutCts.Token);

                result = await ReasonCoreAsync(query, config, linkedCts.Token);
            }
            else
            {
                result = await ReasonCoreAsync(query, config, cancellationToken);
            }

            result.Success = true;
        }
        catch (OperationCanceledException)
        {
            result.Success = false;
            result.ErrorMessage = $"Reasoning timeout after {config.MaxReasoningTimeSeconds} seconds";
            AppendTrace($"ERROR: {result.ErrorMessage}");
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.ErrorMessage = $"Reasoning failed: {ex.Message}";
            AppendTrace($"ERROR: {ex.Message}");
        }
        finally
        {
            stopwatch.Stop();
            result.TotalDuration = stopwatch.Elapsed;
            result.Metrics["total_duration_ms"] = stopwatch.ElapsedMilliseconds;
        }

        return result;
    }

    /// <summary>
    /// Core reasoning logic to be implemented by derived strategies.
    /// </summary>
    /// <param name="query">The problem or question to reason about.</param>
    /// <param name="config">Configuration options for the reasoning process.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A reasoning result containing the answer and reasoning process.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is where the actual reasoning happens - and it's different
    /// for each strategy. Derived classes must implement this method with their specific approach:
    /// - ChainOfThoughtStrategy: Linear step-by-step reasoning
    /// - TreeOfThoughtsStrategy: Explores multiple paths in a tree
    /// - SelfConsistencyStrategy: Multiple attempts with voting
    ///
    /// The base class handles setup, timing, and error handling; derived classes focus on
    /// implementing their unique reasoning algorithm.
    /// </para>
    /// </remarks>
    protected abstract Task<ReasoningResult<T>> ReasonCoreAsync(
        string query,
        ReasoningConfig config,
        CancellationToken cancellationToken);

    /// <summary>
    /// Appends a message to the reasoning trace.
    /// </summary>
    /// <param name="message">The message to append.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is like writing in your reasoning notebook. Call this method
    /// whenever you want to record what's happening during reasoning. For example:
    /// - "Generating reasoning steps..."
    /// - "Evaluating thought quality: 0.85"
    /// - "Verification passed for step 3"
    ///
    /// All these traces get included in the final result for transparency and debugging.
    /// </para>
    /// </remarks>
    protected void AppendTrace(string message)
    {
        lock (_traceLock)
        {
            _reasoningTrace.AppendLine($"[{DateTime.UtcNow:HH:mm:ss.fff}] {message}");
        }
    }

    /// <summary>
    /// Clears the reasoning trace, starting fresh.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This erases the reasoning notebook, starting with a clean page.
    /// Called at the start of each new reasoning task to avoid mixing up traces from different problems.
    /// </para>
    /// </remarks>
    protected void ClearTrace()
    {
        lock (_traceLock)
        {
            _reasoningTrace.Clear();
        }
    }

    /// <summary>
    /// Finds a tool by its name.
    /// </summary>
    /// <param name="toolName">The name of the tool to find.</param>
    /// <returns>The tool with the specified name, or null if not found.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This searches through the available tools for one with a specific name.
    /// The search is case-insensitive, so "Calculator", "calculator", and "CALCULATOR" all match.
    ///
    /// Returns null if the tool doesn't exist, so always check before using!
    /// </para>
    /// </remarks>
    protected ITool? FindTool(string toolName)
    {
        return _tools.FirstOrDefault(t =>
            t.Name.Equals(toolName, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Gets a formatted description of all available tools.
    /// </summary>
    /// <returns>A string describing all tools and their capabilities.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a text description of all available tools that can be
    /// included in prompts to the language model. This tells the AI what tools it can use.
    ///
    /// Example output:
    /// "Available tools:
    /// - Calculator: Performs mathematical calculations
    /// - WebSearch: Searches the internet for information"
    /// </para>
    /// </remarks>
    protected string GetToolDescriptions()
    {
        if (!_tools.Any())
        {
            return "No tools available.";
        }

        var sb = new System.Text.StringBuilder("Available tools:\n");
        foreach (var tool in _tools)
        {
            sb.AppendLine($"- {tool.Name}: {tool.Description}");
        }
        return sb.ToString();
    }

    /// <summary>
    /// Executes a tool with the given input.
    /// </summary>
    /// <param name="toolName">Name of the tool to execute.</param>
    /// <param name="input">Input for the tool.</param>
    /// <returns>The tool's output, or an error message if execution failed.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This runs a tool and returns its result. For example:
    /// - ExecuteTool("Calculator", "15 * 240") → "3600"
    /// - ExecuteTool("WebSearch", "capital of France") → "Paris"
    ///
    /// If the tool doesn't exist or fails to execute, returns an error message instead of throwing.
    /// This keeps the reasoning process running even if one tool fails.
    /// </para>
    /// </remarks>
    protected string ExecuteTool(string toolName, string input)
    {
        var tool = FindTool(toolName);
        if (tool == null)
        {
            return $"Error: Tool '{toolName}' not found. Available tools: {string.Join(", ", _tools.Select(t => t.Name))}";
        }

        try
        {
            var result = tool.Execute(input);
            AppendTrace($"Tool '{toolName}' executed successfully");
            return result;
        }
        catch (Exception ex)
        {
            // Rethrow critical exceptions
            if (ex is OutOfMemoryException || ex is StackOverflowException)
                throw;

            var errorMsg = $"Error executing tool '{toolName}': {ex.Message}";
            AppendTrace(errorMsg);
            return errorMsg;
        }
    }

    /// <summary>
    /// Validates that a reasoning configuration is valid.
    /// </summary>
    /// <param name="config">The configuration to validate.</param>
    /// <exception cref="ArgumentException">Thrown when configuration is invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This checks that all configuration settings make sense.
    /// For example:
    /// - MaxSteps must be positive
    /// - Scores must be between 0 and 1
    /// - Beam width must be positive
    ///
    /// Call this in derived classes to validate config before using it.
    /// </para>
    /// </remarks>
    protected virtual void ValidateConfig(ReasoningConfig config)
    {
        if (config.MaxSteps < 1)
            throw new ArgumentException("MaxSteps must be at least 1", nameof(config));

        if (config.ExplorationDepth < 1)
            throw new ArgumentException("ExplorationDepth must be at least 1", nameof(config));

        if (config.BranchingFactor < 1)
            throw new ArgumentException("BranchingFactor must be at least 1", nameof(config));

        if (config.NumSamples < 1)
            throw new ArgumentException("NumSamples must be at least 1", nameof(config));

        if (config.Temperature < 0.0)
            throw new ArgumentException("Temperature cannot be negative", nameof(config));

        if (config.BeamWidth < 1)
            throw new ArgumentException("BeamWidth must be at least 1", nameof(config));

        if (config.VerificationThreshold < 0.0 || config.VerificationThreshold > 1.0)
            throw new ArgumentException("VerificationThreshold must be between 0.0 and 1.0", nameof(config));

        if (config.MaxRefinementAttempts < 0)
            throw new ArgumentException("MaxRefinementAttempts cannot be negative", nameof(config));

        if (config.ComputeScalingFactor <= 0.0)
            throw new ArgumentException("ComputeScalingFactor must be positive", nameof(config));

        if (config.MaxReasoningTimeSeconds < 0)
            throw new ArgumentException("MaxReasoningTimeSeconds cannot be negative", nameof(config));
    }
}
