namespace AiDotNet.PromptEngineering.Compression;

/// <summary>
/// Contains the result of a prompt compression operation including metrics.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates the compressed prompt along with detailed metrics
/// about the compression operation, including token counts, compression ratio,
/// and estimated cost savings.
/// </para>
/// <para><b>For Beginners:</b> This is the result of compressing a prompt, with before/after stats.
///
/// When you compress a prompt, you want to know:
/// - What's the compressed text?
/// - How much shorter is it?
/// - How much money did we save?
///
/// Example:
/// <code>
/// var result = compressor.CompressWithMetrics(longPrompt, options);
///
/// Console.WriteLine($"Before: {result.OriginalTokenCount} tokens");
/// Console.WriteLine($"After: {result.CompressedTokenCount} tokens");
/// Console.WriteLine($"Saved: {result.TokensSaved} tokens ({result.CompressionRatio:P0})");
/// Console.WriteLine($"Cost savings: ${result.EstimatedCostSavings}");
/// Console.WriteLine($"Compressed prompt: {result.CompressedPrompt}");
/// </code>
/// </para>
/// </remarks>
public class CompressionResult
{
    /// <summary>
    /// Gets or sets the original prompt before compression.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The prompt you started with, before any changes.
    /// Stored so you can compare or revert if needed.
    /// </para>
    /// </remarks>
    public string OriginalPrompt { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the compressed prompt.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The shorter version of your prompt.
    /// This is what you'd actually send to the AI model.
    /// </para>
    /// </remarks>
    public string CompressedPrompt { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the token count of the original prompt.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many tokens the original prompt had.
    /// This is your "before" measurement.
    /// </para>
    /// </remarks>
    public int OriginalTokenCount { get; set; }

    /// <summary>
    /// Gets or sets the token count of the compressed prompt.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many tokens after compression.
    /// This is your "after" measurement.
    /// </para>
    /// </remarks>
    public int CompressedTokenCount { get; set; }

    /// <summary>
    /// Gets the number of tokens saved by compression.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many tokens were removed.
    /// OriginalTokenCount - CompressedTokenCount
    /// </para>
    /// </remarks>
    public int TokensSaved => OriginalTokenCount - CompressedTokenCount;

    /// <summary>
    /// Gets the compression ratio (0.0 to 1.0, where higher means more compression).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Calculated as (OriginalTokenCount - CompressedTokenCount) / OriginalTokenCount.
    /// A ratio of 0.3 means 30% of tokens were removed.
    /// </para>
    /// <para><b>For Beginners:</b> What percentage of the prompt was removed.
    /// - 0.0 = No compression (same size)
    /// - 0.5 = 50% smaller
    /// - 1.0 = Everything removed (shouldn't happen!)
    /// </para>
    /// </remarks>
    public double CompressionRatio => OriginalTokenCount > 0
        ? (double)(OriginalTokenCount - CompressedTokenCount) / OriginalTokenCount
        : 0.0;

    /// <summary>
    /// Gets or sets the estimated cost savings in USD.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Based on the token reduction and current API pricing for the target model.
    /// Useful for ROI calculations on compression efforts.
    /// </para>
    /// <para><b>For Beginners:</b> How much money you saved by using less tokens.
    /// If compression saved 100 tokens on GPT-4 (~$0.003 per 1K tokens input),
    /// savings would be about $0.0003.
    /// </para>
    /// </remarks>
    public decimal EstimatedCostSavings { get; set; }

    /// <summary>
    /// Gets or sets the compression method used.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How the prompt was compressed.
    /// Examples: "RedundancyRemoval", "Summarization", "Caching"
    /// </para>
    /// </remarks>
    public string CompressionMethod { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets any warnings generated during compression.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Warnings about potential issues with the compression, such as
    /// possible loss of important information or reaching minimum length.
    /// </para>
    /// <para><b>For Beginners:</b> Problems that occurred during compression.
    /// For example: "Could not compress further without losing meaning"
    /// </para>
    /// </remarks>
    public IReadOnlyList<string> Warnings { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the timestamp when compression was performed.
    /// </summary>
    public DateTime CompressedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets whether the compression was successful (reduced token count).
    /// </summary>
    public bool IsSuccessful => CompressedTokenCount < OriginalTokenCount;
}

/// <summary>
/// Options for controlling prompt compression behavior.
/// </summary>
/// <remarks>
/// <para>
/// Configures how aggressive compression should be, what techniques to use,
/// and constraints on the output.
/// </para>
/// <para><b>For Beginners:</b> Settings for how to compress your prompt.
///
/// Example:
/// <code>
/// var options = new CompressionOptions
/// {
///     TargetReduction = 0.3,  // Try to reduce by 30%
///     PreserveVariables = true,  // Don't remove {placeholders}
///     MinTokenCount = 50  // Don't compress below 50 tokens
/// };
/// </code>
/// </para>
/// </remarks>
public class CompressionOptions
{
    /// <summary>
    /// Gets or sets the target reduction ratio (0.0 to 1.0).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The desired percentage of tokens to remove. A value of 0.3 means
    /// try to remove 30% of tokens. The compressor will try to achieve this
    /// target but may not always succeed.
    /// </para>
    /// <para><b>For Beginners:</b> How much smaller you want the prompt.
    /// - 0.2 = Try to make it 20% smaller
    /// - 0.5 = Try to make it 50% smaller
    /// Higher values = more aggressive compression
    /// </para>
    /// </remarks>
    public double TargetReduction { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the maximum number of tokens in the output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// If set, compression will try to get the output below this token count.
    /// Useful when you need to fit within a specific context window.
    /// </para>
    /// <para><b>For Beginners:</b> The maximum tokens allowed after compression.
    /// Set this if you have a hard limit to fit within.
    /// </para>
    /// </remarks>
    public int? MaxTokens { get; set; }

    /// <summary>
    /// Gets or sets the minimum number of tokens in the output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A safety limit to prevent over-compression that would lose essential content.
    /// </para>
    /// <para><b>For Beginners:</b> Don't compress below this many tokens.
    /// Prevents the prompt from becoming too short to be useful.
    /// </para>
    /// </remarks>
    public int MinTokenCount { get; set; } = 10;

    /// <summary>
    /// Gets or sets whether to preserve template variables during compression.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, placeholders like {variable_name} will not be modified or removed.
    /// </para>
    /// <para><b>For Beginners:</b> Keep {placeholders} intact.
    /// You don't want compression to break your template!
    /// </para>
    /// </remarks>
    public bool PreserveVariables { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to preserve code blocks during compression.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, content within code blocks (```) will not be modified.
    /// Important for prompts that include code examples.
    /// </para>
    /// <para><b>For Beginners:</b> Keep code examples unchanged.
    /// You don't want compression to break your code!
    /// </para>
    /// </remarks>
    public bool PreserveCodeBlocks { get; set; } = true;

    /// <summary>
    /// Gets or sets the model name for accurate token counting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Different models tokenize differently. Specifying the target model
    /// ensures accurate token counts for that model.
    /// </para>
    /// <para><b>For Beginners:</b> Which AI model you'll use this prompt with.
    /// Helps count tokens accurately for your specific model.
    /// </para>
    /// </remarks>
    public string ModelName { get; set; } = "gpt-4";

    /// <summary>
    /// Gets default compression options with moderate settings.
    /// </summary>
    public static CompressionOptions Default => new();

    /// <summary>
    /// Gets aggressive compression options for maximum reduction.
    /// </summary>
    public static CompressionOptions Aggressive => new()
    {
        TargetReduction = 0.5,
        MinTokenCount = 20
    };

    /// <summary>
    /// Gets conservative compression options that preserve more content.
    /// </summary>
    public static CompressionOptions Conservative => new()
    {
        TargetReduction = 0.1,
        PreserveVariables = true,
        PreserveCodeBlocks = true
    };
}
