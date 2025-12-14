using AiDotNet.PromptEngineering.Analysis;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for analyzing prompts before sending them to language models.
/// </summary>
/// <remarks>
/// <para>
/// A prompt analyzer computes metrics and validates prompts to help developers understand
/// and optimize their prompts before incurring API costs. Analysis includes token counting,
/// cost estimation, complexity measurement, and issue detection.
/// </para>
/// <para><b>For Beginners:</b> A prompt analyzer is like a spell-checker and cost calculator for your prompts.
///
/// Before sending a prompt to an AI model (which costs money), the analyzer tells you:
/// - How many tokens it uses (tokens = cost)
/// - Estimated API cost in dollars
/// - How complex the prompt is
/// - Any potential problems (missing variables, too long, etc.)
///
/// Example workflow:
/// <code>
/// var analyzer = new TokenCountAnalyzer();
/// var metrics = analyzer.Analyze("Translate this text...");
///
/// Console.WriteLine($"Tokens: {metrics.TokenCount}");
/// Console.WriteLine($"Cost: ${metrics.EstimatedCost}");
///
/// if (metrics.TokenCount > 4000)
/// {
///     Console.WriteLine("Warning: Prompt is very long, consider compression");
/// }
/// </code>
///
/// Benefits:
/// - Cost control: Know costs before making API calls
/// - Optimization: Find prompts that are too long or complex
/// - Debugging: Catch issues before they cause errors
/// - Budgeting: Track and forecast API spending
/// </para>
/// </remarks>
public interface IPromptAnalyzer
{
    /// <summary>
    /// Analyzes a prompt and returns detailed metrics.
    /// </summary>
    /// <param name="prompt">The prompt string to analyze.</param>
    /// <returns>A PromptMetrics object containing analysis results.</returns>
    /// <remarks>
    /// <para>
    /// Performs comprehensive analysis of the prompt including token counting,
    /// cost estimation, complexity scoring, and pattern detection.
    /// </para>
    /// <para><b>For Beginners:</b> This examines your prompt and tells you everything about it.
    ///
    /// Example:
    /// <code>
    /// var metrics = analyzer.Analyze("Summarize this article about climate change...");
    ///
    /// // metrics.TokenCount: 150 (how many tokens)
    /// // metrics.EstimatedCost: $0.003 (cost at current rates)
    /// // metrics.ComplexityScore: 0.4 (0-1, higher = more complex)
    /// // metrics.VariableCount: 0 (number of {variables})
    /// // metrics.DetectedPatterns: ["summarization", "instruction"]
    /// </code>
    /// </para>
    /// </remarks>
    PromptMetrics Analyze(string prompt);

    /// <summary>
    /// Analyzes a prompt asynchronously for use in async workflows.
    /// </summary>
    /// <param name="prompt">The prompt string to analyze.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A task that resolves to PromptMetrics.</returns>
    /// <remarks>
    /// <para>
    /// Async version of Analyze for non-blocking analysis, useful when analyzing
    /// many prompts or when analysis involves external calls (e.g., to a tokenizer API).
    /// </para>
    /// <para><b>For Beginners:</b> Same as Analyze, but doesn't block your program.
    /// Use this when analyzing many prompts at once or in web applications.
    /// </para>
    /// </remarks>
    Task<PromptMetrics> AnalyzeAsync(string prompt, CancellationToken cancellationToken = default);

    /// <summary>
    /// Validates a prompt for potential issues.
    /// </summary>
    /// <param name="prompt">The prompt string to validate.</param>
    /// <param name="options">Options controlling validation behavior.</param>
    /// <returns>A collection of detected issues, empty if no issues found.</returns>
    /// <remarks>
    /// <para>
    /// Checks the prompt for common problems such as missing placeholders,
    /// excessive length, potential injection vulnerabilities, and format issues.
    /// </para>
    /// <para><b>For Beginners:</b> This looks for problems in your prompt.
    ///
    /// Example issues it might find:
    /// - "Prompt exceeds maximum length for GPT-4 (8192 tokens)"
    /// - "Variable {user_input} contains potential prompt injection"
    /// - "Missing closing brace for variable {name"
    /// - "Prompt is empty or whitespace-only"
    ///
    /// Use this before sending prompts to catch problems early:
    /// <code>
    /// var issues = analyzer.ValidatePrompt(prompt, ValidationOptions.Strict);
    /// if (issues.Any())
    /// {
    ///     foreach (var issue in issues)
    ///     {
    ///         Console.WriteLine($"[{issue.Severity}] {issue.Message}");
    ///     }
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    IEnumerable<PromptIssue> ValidatePrompt(string prompt, ValidationOptions? options = null);

    /// <summary>
    /// Gets the name of this analyzer implementation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A human-readable identifier for this analyzer, useful for logging
    /// and debugging which analyzer is being used.
    /// </para>
    /// <para><b>For Beginners:</b> The name of this specific analyzer.
    /// Examples: "TokenCountAnalyzer", "GPT4CostEstimator", "ClaudeAnalyzer"
    /// </para>
    /// </remarks>
    string Name { get; }
}
