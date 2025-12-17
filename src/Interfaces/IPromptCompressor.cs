using AiDotNet.PromptEngineering.Compression;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for compressing prompts to reduce token counts and API costs.
/// </summary>
/// <remarks>
/// <para>
/// A prompt compressor reduces the length of prompts while preserving their semantic meaning.
/// This is valuable for reducing API costs, fitting within context windows, and optimizing
/// performance. Different compression strategies include redundancy removal, summarization,
/// and caching-based approaches.
/// </para>
/// <para><b>For Beginners:</b> A prompt compressor makes your prompts shorter without losing meaning.
///
/// Why compress prompts?
/// - Save money: Shorter prompts = fewer tokens = lower costs
/// - Fit limits: Some models have maximum token limits
/// - Faster: Shorter prompts process faster
///
/// Example:
/// <code>
/// var compressor = new RedundancyCompressor();
///
/// string longPrompt = @"
///     Please analyze the following document and provide a summary.
///     The document that you need to analyze is provided below.
///     When you analyze the document, focus on the main points.
///     Here is the document to analyze: [document text]";
///
/// var result = compressor.CompressWithMetrics(longPrompt, new CompressionOptions { TargetReduction = 0.3 });
///
/// Console.WriteLine($"Original: {result.OriginalTokenCount} tokens");
/// Console.WriteLine($"Compressed: {result.CompressedTokenCount} tokens");
/// Console.WriteLine($"Savings: {result.CompressionRatio:P0}");
///
/// // Output:
/// // Original: 80 tokens
/// // Compressed: 45 tokens
/// // Savings: 44%
/// </code>
///
/// The compressed version might be:
/// "Analyze this document and summarize the main points: [document text]"
/// </para>
/// </remarks>
public interface IPromptCompressor
{
    /// <summary>
    /// Compresses a prompt to reduce its token count.
    /// </summary>
    /// <param name="prompt">The prompt string to compress.</param>
    /// <param name="options">Options controlling compression behavior.</param>
    /// <returns>The compressed prompt string.</returns>
    /// <remarks>
    /// <para>
    /// Applies compression techniques to reduce the prompt's token count while
    /// preserving its semantic meaning and effectiveness for the target task.
    /// </para>
    /// <para><b>For Beginners:</b> This takes your prompt and makes it shorter.
    ///
    /// Example:
    /// <code>
    /// var compressed = compressor.Compress(
    ///     "Please kindly help me to translate the following text from English to Spanish",
    ///     new CompressionOptions());
    ///
    /// // Result: "Translate from English to Spanish"
    /// </code>
    ///
    /// The compressor removes:
    /// - Redundant words ("please kindly help me to")
    /// - Unnecessary phrases ("the following")
    /// - Verbose constructions
    /// </para>
    /// </remarks>
    string Compress(string prompt, CompressionOptions? options = null);

    /// <summary>
    /// Compresses a prompt and returns detailed metrics about the compression.
    /// </summary>
    /// <param name="prompt">The prompt string to compress.</param>
    /// <param name="options">Options controlling compression behavior.</param>
    /// <returns>A CompressionResult containing the compressed prompt and metrics.</returns>
    /// <remarks>
    /// <para>
    /// In addition to compressing the prompt, this method returns detailed metrics
    /// including original and compressed token counts, compression ratio, and
    /// information about what was changed.
    /// </para>
    /// <para><b>For Beginners:</b> Like Compress, but also tells you what changed.
    ///
    /// Example:
    /// <code>
    /// var result = compressor.CompressWithMetrics(longPrompt, options);
    ///
    /// Console.WriteLine($"Original tokens: {result.OriginalTokenCount}");
    /// Console.WriteLine($"Compressed tokens: {result.CompressedTokenCount}");
    /// Console.WriteLine($"Tokens saved: {result.TokensSaved}");
    /// Console.WriteLine($"Compression ratio: {result.CompressionRatio:P0}");
    /// Console.WriteLine($"Estimated savings: ${result.EstimatedCostSavings}");
    /// </code>
    ///
    /// This is useful for:
    /// - Tracking cost savings
    /// - Debugging compression issues
    /// - Reporting to stakeholders
    /// </para>
    /// </remarks>
    CompressionResult CompressWithMetrics(string prompt, CompressionOptions? options = null);

    /// <summary>
    /// Compresses a prompt asynchronously.
    /// </summary>
    /// <param name="prompt">The prompt string to compress.</param>
    /// <param name="options">Options controlling compression behavior.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A task that resolves to the compressed prompt string.</returns>
    /// <remarks>
    /// <para>
    /// Async version of Compress for non-blocking compression, particularly useful
    /// for summarization-based compression that may call external LLM APIs.
    /// </para>
    /// <para><b>For Beginners:</b> Same as Compress, but doesn't block your program.
    /// Important for compressors that use AI to summarize (which takes time).
    /// </para>
    /// </remarks>
    Task<string> CompressAsync(string prompt, CompressionOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the name of this compressor implementation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A human-readable identifier for this compressor, useful for logging
    /// and debugging which compressor is being used.
    /// </para>
    /// <para><b>For Beginners:</b> The name of this specific compressor.
    /// Examples: "RedundancyCompressor", "SummarizationCompressor", "CachingCompressor"
    /// </para>
    /// </remarks>
    string Name { get; }
}
