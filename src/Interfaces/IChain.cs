namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for chains that compose multiple language model operations.
/// </summary>
/// <typeparam name="TInput">The type of input accepted by the chain.</typeparam>
/// <typeparam name="TOutput">The type of output produced by the chain.</typeparam>
/// <remarks>
/// <para>
/// A chain orchestrates multiple language model calls, tools, and transformations into a cohesive workflow.
/// Chains can be sequential, conditional, parallel, or implement complex patterns like map-reduce or routing.
/// </para>
/// <para><b>For Beginners:</b> A chain connects multiple steps into a complete workflow.
///
/// Think of a chain like a recipe:
/// - Each step does something specific
/// - Steps happen in a particular order (or in parallel)
/// - Output from one step can feed into another
/// - The final result is a complete dish
///
/// Example - Customer support chain:
/// Input: Customer email
/// Step 1: Classify email type (question/complaint/feedback)
/// Step 2: Route to appropriate handler
/// Step 3: Generate response
/// Step 4: Add personalization
/// Output: Personalized response
///
/// Chains make complex workflows:
/// - Modular: Each step is separate and testable
/// - Reusable: Steps can be used in different chains
/// - Maintainable: Easy to modify or extend
/// - Understandable: Clear flow from input to output
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("Chain")]
public interface IChain<TInput, TOutput>
{
    /// <summary>
    /// Executes the chain with the provided input.
    /// </summary>
    /// <param name="input">The input to process.</param>
    /// <returns>The chain's output result.</returns>
    /// <remarks>
    /// <para>
    /// Runs the complete chain workflow, processing the input through all steps and
    /// returning the final output. The exact behavior depends on the chain type
    /// (sequential, conditional, parallel, etc.).
    /// </para>
    /// <para><b>For Beginners:</b> This runs your entire workflow.
    ///
    /// Example - Text summarization chain:
    /// Input: Long article (10,000 words)
    ///
    /// Internal steps (handled automatically):
    /// 1. Split article into chunks
    /// 2. Summarize each chunk
    /// 3. Combine chunk summaries
    /// 4. Generate final summary
    ///
    /// Output: Concise summary (200 words)
    ///
    /// You just call:
    /// var summary = chain.Run(longArticle);
    ///
    /// The chain handles all the complexity internally.
    /// </para>
    /// </remarks>
    TOutput Run(TInput input);

    /// <summary>
    /// Executes the chain asynchronously with the provided input.
    /// </summary>
    /// <param name="input">The input to process.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A task representing the asynchronous operation with the chain's output.</returns>
    /// <remarks>
    /// <para>
    /// Asynchronous version of Run() that allows for cancellation and doesn't block the calling thread.
    /// Particularly useful for chains involving API calls or long-running operations.
    /// </para>
    /// <para><b>For Beginners:</b> Same as Run(), but doesn't freeze your program while waiting.
    ///
    /// Synchronous (Run):
    /// - Starts the chain
    /// - Waits for completion
    /// - Your program is frozen during this time
    /// - Returns result
    ///
    /// Asynchronous (RunAsync):
    /// - Starts the chain
    /// - Your program continues working
    /// - You can cancel if needed
    /// - Returns result when ready
    ///
    /// Use RunAsync when:
    /// - Chain involves API calls (network delays)
    /// - Processing takes a long time
    /// - Building responsive UIs
    /// - Need ability to cancel
    ///
    /// Example:
    /// ```csharp
    /// var cancellation = new CancellationTokenSource();
    ///
    /// // Start chain (doesn't block)
    /// var task = chain.RunAsync(input, cancellation.Token);
    ///
    /// // Do other work while chain runs
    /// UpdateUI("Processing...");
    ///
    /// // If user cancels
    /// if (userClickedCancel)
    /// {
    ///     cancellation.Cancel();
    /// }
    ///
    /// // Wait for result
    /// var result = await task;
    /// ```
    /// </para>
    /// </remarks>
    Task<TOutput> RunAsync(TInput input, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the name of this chain.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A descriptive name for the chain, useful for logging, debugging, and documentation.
    /// </para>
    /// <para><b>For Beginners:</b> The chain's identifier for debugging and logging.
    ///
    /// Examples:
    /// - "CustomerSupportChain"
    /// - "DocumentSummarization"
    /// - "SentimentAnalysisPipeline"
    ///
    /// Helps with:
    /// - Debugging: See which chain is running in logs
    /// - Monitoring: Track performance by chain name
    /// - Documentation: Understand system architecture
    /// </para>
    /// </remarks>
    string Name { get; }

    /// <summary>
    /// Gets the description of what this chain does.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A human-readable description explaining the chain's purpose and behavior.
    /// </para>
    /// <para><b>For Beginners:</b> Explains what the chain does.
    ///
    /// Example:
    /// Name: "DocumentQAChain"
    /// Description: "Answers questions about documents by retrieving relevant sections
    ///              and generating grounded responses with citations."
    ///
    /// This helps:
    /// - Team members understand the chain's purpose
    /// - Decide which chain to use for a task
    /// - Generate documentation automatically
    /// </para>
    /// </remarks>
    string Description { get; }

    /// <summary>
    /// Validates that the input is appropriate for this chain.
    /// </summary>
    /// <param name="input">The input to validate.</param>
    /// <returns>True if the input is valid; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// Checks whether the provided input meets the chain's requirements before execution.
    /// This helps catch errors early and provide meaningful error messages.
    /// </para>
    /// <para><b>For Beginners:</b> Checks if the input is acceptable before running the chain.
    ///
    /// Example - Email response chain:
    /// Requirements: Input must be a non-empty string, under 10,000 characters
    ///
    /// Validate("") → False (empty)
    /// Validate(null) → False (null)
    /// Validate("Hello, I have a question...") → True
    /// Validate([11,000 character string]) → False (too long)
    ///
    /// Benefits:
    /// - Fail fast: Catch problems before expensive operations
    /// - Clear errors: "Input too long" vs cryptic runtime error
    /// - Better UX: Immediate feedback to users
    /// </para>
    /// </remarks>
    bool ValidateInput(TInput input);
}
