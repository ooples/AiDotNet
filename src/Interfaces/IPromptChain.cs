namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for executing multi-step prompt workflows.
/// </summary>
/// <remarks>
/// <para>
/// A prompt chain enables complex multi-step workflows where the output of one step
/// becomes the input to the next. Chains support sequential execution, parallel execution,
/// conditional branching, and map-reduce patterns for processing multiple items.
/// </para>
/// <para><b>For Beginners:</b> A prompt chain is like an assembly line for AI tasks.
///
/// Instead of doing everything in one big prompt, you break it into steps:
/// <code>
/// // Example: Translate and Summarize
/// Step 1: Translate document from Spanish to English
/// Step 2: Summarize the translated document
/// Step 3: Extract key points as bullet points
/// </code>
///
/// Each step takes the previous step's output as input.
///
/// Benefits of chains:
/// - Simpler prompts (each does one thing well)
/// - Better quality (specialized prompts perform better)
/// - Easier debugging (you can inspect intermediate results)
/// - Flexible workflows (add/remove/modify steps)
///
/// Chain types:
/// - Sequential: Step 1 → Step 2 → Step 3
/// - Parallel: Steps 1, 2, 3 run simultaneously, results merged
/// - Conditional: If X then Step A, else Step B
/// - Map-Reduce: Process many items in parallel, then combine
/// </para>
/// </remarks>
public interface IPromptChain
{
    /// <summary>
    /// Executes the chain synchronously with the given input.
    /// </summary>
    /// <param name="input">The initial input to the chain.</param>
    /// <returns>The result of executing all steps in the chain.</returns>
    /// <remarks>
    /// <para>
    /// Executes each step in the chain in sequence (or in parallel if configured),
    /// passing the output of each step as input to the next.
    /// </para>
    /// <para><b>For Beginners:</b> Runs all the steps and returns the final result.
    ///
    /// Example:
    /// <code>
    /// var chain = new SequentialChain()
    ///     .AddStep("Translate to English", translatePrompt)
    ///     .AddStep("Summarize", summarizePrompt);
    ///
    /// string result = chain.Execute("Documento en español...");
    /// // result = "Summary: This document discusses..."
    /// </code>
    /// </para>
    /// </remarks>
    ChainResult Execute(string input);

    /// <summary>
    /// Executes the chain asynchronously with the given input.
    /// </summary>
    /// <param name="input">The initial input to the chain.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A task that resolves to the chain execution result.</returns>
    /// <remarks>
    /// <para>
    /// Async version of Execute for non-blocking execution. Essential for chains
    /// that make API calls to language models.
    /// </para>
    /// <para><b>For Beginners:</b> Same as Execute but doesn't block your program.
    ///
    /// Use this when:
    /// - Running in a web application
    /// - Processing many items in parallel
    /// - Making actual API calls to language models
    ///
    /// Example:
    /// <code>
    /// var result = await chain.ExecuteAsync("Input text...");
    /// </code>
    /// </para>
    /// </remarks>
    Task<ChainResult> ExecuteAsync(string input, CancellationToken cancellationToken = default);

    /// <summary>
    /// Executes the chain with named inputs (for chains with multiple entry points).
    /// </summary>
    /// <param name="inputs">A dictionary of named inputs.</param>
    /// <returns>The result of executing the chain.</returns>
    /// <remarks>
    /// <para>
    /// Some chains accept multiple inputs by name. This method allows passing
    /// all inputs at once as a dictionary.
    /// </para>
    /// <para><b>For Beginners:</b> For chains that need multiple pieces of information.
    ///
    /// Example - Comparison chain:
    /// <code>
    /// var chain = new ComparisonChain(); // Compares two documents
    ///
    /// var inputs = new Dictionary&lt;string, string&gt;
    /// {
    ///     ["document1"] = "First document text...",
    ///     ["document2"] = "Second document text..."
    /// };
    ///
    /// var result = chain.Execute(inputs);
    /// // result = "Similarities: ... Differences: ..."
    /// </code>
    /// </para>
    /// </remarks>
    ChainResult Execute(IDictionary<string, string> inputs);

    /// <summary>
    /// Executes the chain with named inputs asynchronously.
    /// </summary>
    /// <param name="inputs">A dictionary of named inputs.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A task that resolves to the chain execution result.</returns>
    Task<ChainResult> ExecuteAsync(IDictionary<string, string> inputs, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the steps in this chain.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns all steps in the chain in execution order. Useful for inspection
    /// and debugging.
    /// </para>
    /// <para><b>For Beginners:</b> Shows what steps are in the chain.
    ///
    /// Example:
    /// <code>
    /// foreach (var step in chain.Steps)
    /// {
    ///     Console.WriteLine($"Step: {step.Name}");
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    IReadOnlyList<IChainStep> Steps { get; }

    /// <summary>
    /// Gets the name of this chain.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A human-readable identifier for this chain. Useful for logging and debugging.
    /// </para>
    /// </remarks>
    string Name { get; }
}

/// <summary>
/// Represents the result of executing a prompt chain.
/// </summary>
/// <remarks>
/// <para>
/// Contains the final output, intermediate results from each step, timing information,
/// and any errors encountered during execution.
/// </para>
/// <para><b>For Beginners:</b> This contains everything about what happened during chain execution.
///
/// You can see:
/// - The final result
/// - What each step produced
/// - How long each step took
/// - Any errors that occurred
///
/// Example:
/// <code>
/// var result = await chain.ExecuteAsync("Input...");
///
/// Console.WriteLine($"Final: {result.FinalOutput}");
/// Console.WriteLine($"Total time: {result.TotalDuration}ms");
///
/// foreach (var step in result.StepResults)
/// {
///     Console.WriteLine($"Step {step.StepName}: {step.Duration}ms");
/// }
/// </code>
/// </para>
/// </remarks>
public class ChainResult
{
    /// <summary>
    /// Gets or sets the final output of the chain.
    /// </summary>
    public string FinalOutput { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets whether the chain executed successfully.
    /// </summary>
    public bool IsSuccessful { get; set; }

    /// <summary>
    /// Gets or sets any error that occurred during execution.
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Gets or sets the results from each step in the chain.
    /// </summary>
    public IReadOnlyList<StepResult> StepResults { get; set; } = new List<StepResult>();

    /// <summary>
    /// Gets or sets the total execution duration in milliseconds.
    /// </summary>
    public long TotalDurationMs { get; set; }

    /// <summary>
    /// Gets or sets intermediate outputs keyed by step name.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Allows accessing the output of any step by name. Useful for debugging
    /// and for chains where intermediate results are needed.
    /// </para>
    /// </remarks>
    public IReadOnlyDictionary<string, string> IntermediateOutputs { get; set; } = new Dictionary<string, string>();
}

/// <summary>
/// Represents the result of executing a single step in a chain.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Information about one step's execution.
///
/// Includes:
/// - What the step produced (output)
/// - How long it took (duration)
/// - Any errors that occurred
/// - Token usage (if available)
/// </para>
/// </remarks>
public class StepResult
{
    /// <summary>
    /// Gets or sets the name of the step.
    /// </summary>
    public string StepName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the input to this step.
    /// </summary>
    public string Input { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the output from this step.
    /// </summary>
    public string Output { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets whether this step executed successfully.
    /// </summary>
    public bool IsSuccessful { get; set; }

    /// <summary>
    /// Gets or sets any error message from this step.
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Gets or sets the execution duration in milliseconds.
    /// </summary>
    public long DurationMs { get; set; }

    /// <summary>
    /// Gets or sets the token count used by this step (if applicable).
    /// </summary>
    public int? TokenCount { get; set; }
}

/// <summary>
/// Defines a single step within a prompt chain.
/// </summary>
/// <remarks>
/// <para>
/// Each step in a chain processes input and produces output. Steps can be
/// simple prompt executions or complex operations like API calls or transformations.
/// </para>
/// <para><b>For Beginners:</b> A step is one operation in the chain.
///
/// Examples of steps:
/// - "Translate text" - sends to translation API
/// - "Summarize document" - sends to summarization prompt
/// - "Extract keywords" - parses text for keywords
/// - "Format as JSON" - transforms output to JSON
///
/// Each step has:
/// - A name (for identification)
/// - An execute function (what it does)
/// - Input/output handling
/// </para>
/// </remarks>
public interface IChainStep
{
    /// <summary>
    /// Gets the name of this step.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Executes this step with the given input.
    /// </summary>
    /// <param name="input">The input to process.</param>
    /// <returns>The output from this step.</returns>
    string Execute(string input);

    /// <summary>
    /// Executes this step asynchronously.
    /// </summary>
    /// <param name="input">The input to process.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A task that resolves to the output from this step.</returns>
    Task<string> ExecuteAsync(string input, CancellationToken cancellationToken = default);
}
