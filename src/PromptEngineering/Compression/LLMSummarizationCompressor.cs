using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.Compression;

/// <summary>
/// Compressor that uses an LLM to intelligently summarize and compress prompts.
/// </summary>
/// <remarks>
/// <para>
/// This compressor delegates compression to a language model, which can understand context
/// and produce semantically equivalent but shorter versions of prompts. It's the most
/// intelligent form of compression but requires an LLM call.
/// </para>
/// <para><b>For Beginners:</b> Uses AI to make your prompt shorter while keeping the meaning.
///
/// Example:
/// <code>
/// var compressor = new LLMSummarizationCompressor(summarizeFunc);
///
/// string verbose = @"I would like you to help me with a task. The task involves
///     taking the following customer feedback data and analyzing it to identify
///     the main themes and patterns. Please pay special attention to any recurring
///     complaints or suggestions that customers have made. After your analysis,
///     provide a summary of your findings.";
///
/// string compressed = await compressor.CompressAsync(verbose);
/// // Result: "Analyze customer feedback to identify themes, recurring complaints,
/// //          and suggestions. Summarize findings."
/// </code>
///
/// When to use:
/// - For complex prompts where simple pattern matching won't work
/// - When semantic understanding is required
/// - When maximum compression with preserved meaning is needed
/// </para>
/// </remarks>
public class LLMSummarizationCompressor : PromptCompressorBase
{
    private readonly Func<string, CancellationToken, Task<string>>? _summarizeFunc;
    private readonly string _systemPrompt;

    /// <summary>
    /// Initializes a new instance of the LLMSummarizationCompressor class.
    /// </summary>
    /// <param name="summarizeFunc">
    /// Function that calls an LLM to summarize text. Takes the text to compress
    /// and returns the compressed version.
    /// </param>
    /// <param name="systemPrompt">Optional custom system prompt for the compression.</param>
    /// <param name="tokenCounter">Optional custom token counter function.</param>
    public LLMSummarizationCompressor(
        Func<string, CancellationToken, Task<string>>? summarizeFunc = null,
        string? systemPrompt = null,
        Func<string, int>? tokenCounter = null)
        : base("LLMSummarization", tokenCounter)
    {
        _summarizeFunc = summarizeFunc;
        _systemPrompt = systemPrompt ?? DefaultSystemPrompt;
    }

    /// <summary>
    /// Gets the default system prompt used for compression.
    /// </summary>
    private static string DefaultSystemPrompt => @"Compress the following prompt while preserving its meaning and intent.
Rules:
1. Keep all variable placeholders like {variable_name} unchanged
2. Keep all code blocks unchanged
3. Maintain the essential instructions and constraints
4. Remove redundant phrases and verbose language
5. Keep the same tone (formal/informal)
6. Preserve numbered or bulleted lists structure
7. Do not add any commentary - just return the compressed prompt";

    /// <summary>
    /// Gets the compression prompt template.
    /// </summary>
    public string SystemPrompt => _systemPrompt;

    /// <summary>
    /// Compresses the prompt synchronously.
    /// </summary>
    /// <remarks>
    /// If no summarization function is provided, falls back to a simple rule-based compression.
    /// For full LLM-based compression, use CompressAsync.
    /// </remarks>
    protected override string CompressCore(string prompt, CompressionOptions options)
    {
        // If no LLM function provided, use fallback compression
        if (_summarizeFunc == null)
        {
            return FallbackCompress(prompt, options);
        }

        // Try to run async version synchronously (not ideal but necessary for sync interface)
        try
        {
            return CompressAsync(prompt, options, CancellationToken.None).GetAwaiter().GetResult();
        }
        catch (Exception)
        {
            // If async call fails, use fallback
            return FallbackCompress(prompt, options);
        }
    }

    /// <summary>
    /// Compresses the prompt asynchronously using the LLM.
    /// </summary>
    public override async Task<string> CompressAsync(
        string prompt,
        CompressionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var opts = options ?? CompressionOptions.Default;

        if (_summarizeFunc == null)
        {
            return FallbackCompress(prompt, opts);
        }

        // Handle variable preservation
        Dictionary<string, string>? variables = null;
        var workingPrompt = prompt;
        if (opts.PreserveVariables)
        {
            variables = ExtractVariables(workingPrompt);
            // Don't replace - let the LLM see them
        }

        // Handle code block preservation
        Dictionary<string, string>? codeBlocks = null;
        if (opts.PreserveCodeBlocks)
        {
            codeBlocks = ExtractCodeBlocks(workingPrompt);
            workingPrompt = ReplaceCodeBlocksWithPlaceholders(workingPrompt, codeBlocks);
        }

        // Build the compression request
        var compressionRequest = BuildCompressionRequest(workingPrompt, opts);

        // Call the LLM
        var compressed = await _summarizeFunc(compressionRequest, cancellationToken).ConfigureAwait(false);

        // Extract just the compressed prompt from the response
        compressed = ExtractCompressedPrompt(compressed);

        // Restore code blocks
        if (codeBlocks != null)
        {
            compressed = RestoreCodeBlocks(compressed, codeBlocks);
        }

        // Validate that variables were preserved
        if (variables != null && opts.PreserveVariables)
        {
            foreach (var kvp in variables)
            {
                if (!compressed.Contains(kvp.Value))
                {
                    // Variable was removed - add it back in a sensible location
                    compressed = compressed.TrimEnd() + " " + kvp.Value;
                }
            }
        }

        return compressed.Trim();
    }

    /// <summary>
    /// Builds the compression request to send to the LLM.
    /// </summary>
    private string BuildCompressionRequest(string prompt, CompressionOptions options)
    {
        var targetLength = options.MaxTokens.HasValue
            ? $"Target length: approximately {options.MaxTokens.Value} tokens or less."
            : $"Target reduction: approximately {options.TargetReduction:P0} shorter.";

        return $@"{_systemPrompt}

{targetLength}

Prompt to compress:
---
{prompt}
---

Compressed prompt:";
    }

    /// <summary>
    /// Extracts the compressed prompt from the LLM response.
    /// </summary>
    private static string ExtractCompressedPrompt(string response)
    {
        // Handle case where LLM adds explanation before/after
        var lines = response.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

        // Skip any lines that look like meta-commentary
        var promptLines = lines.Where(line =>
            !line.TrimStart().StartsWith("Here") &&
            !line.TrimStart().StartsWith("The compressed") &&
            !line.TrimStart().StartsWith("I've") &&
            !line.TrimStart().StartsWith("Note:") &&
            !line.TrimStart().StartsWith("Compressed:"))
            .ToList();

        return string.Join("\n", promptLines).Trim();
    }

    /// <summary>
    /// Fallback compression using rule-based approach when LLM is not available.
    /// </summary>
    private string FallbackCompress(string prompt, CompressionOptions options)
    {
        // Use a combination of other compressors
        var redundancyCompressor = new RedundancyRemovalCompressor();
        var sentenceCompressor = new SentenceCompressor();

        var compressed = redundancyCompressor.Compress(prompt, options);
        compressed = sentenceCompressor.Compress(compressed, options);

        return compressed;
    }
}
