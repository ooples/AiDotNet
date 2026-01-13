using System.Text.RegularExpressions;
using AiDotNet.Interfaces;

namespace AiDotNet.PromptEngineering.Compression;

/// <summary>
/// Provides a base implementation for prompt compressors with common functionality.
/// </summary>
/// <remarks>
/// <para>
/// This abstract class implements the IPromptCompressor interface and provides common functionality
/// for prompt compression strategies. It handles token counting, validation, and delegates to
/// derived classes for the core compression logic.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all prompt compressors build upon.
///
/// Think of it like a template for making prompts shorter:
/// - It handles common tasks (counting tokens, checking inputs)
/// - Specific compression methods fill in how they compress
/// - This ensures all compressors work consistently
///
/// Derived classes just need to implement CompressCore to define their compression logic.
/// </para>
/// </remarks>
public abstract class PromptCompressorBase : IPromptCompressor
{
    /// <summary>
    /// Regex timeout to prevent ReDoS attacks.
    /// </summary>

    private readonly Func<string, int>? _tokenCounter;

    /// <summary>
    /// Initializes a new instance of the PromptCompressorBase class.
    /// </summary>
    /// <param name="name">The name of this compressor.</param>
    /// <param name="tokenCounter">Optional custom token counter function.</param>
    protected PromptCompressorBase(string name, Func<string, int>? tokenCounter = null)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        _tokenCounter = tokenCounter;
    }

    /// <summary>
    /// Gets the name of this compressor implementation.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Compresses a prompt to reduce its token count.
    /// </summary>
    /// <param name="prompt">The prompt string to compress.</param>
    /// <param name="options">Options controlling compression behavior.</param>
    /// <returns>The compressed prompt string.</returns>
    public string Compress(string prompt, CompressionOptions? options = null)
    {
        ValidatePrompt(prompt);
        var opts = options ?? CompressionOptions.Default;
        return CompressCore(prompt, opts);
    }

    /// <summary>
    /// Compresses a prompt and returns detailed metrics about the compression.
    /// </summary>
    /// <param name="prompt">The prompt string to compress.</param>
    /// <param name="options">Options controlling compression behavior.</param>
    /// <returns>A CompressionResult containing the compressed prompt and metrics.</returns>
    public CompressionResult CompressWithMetrics(string prompt, CompressionOptions? options = null)
    {
        ValidatePrompt(prompt);
        var opts = options ?? CompressionOptions.Default;

        var originalTokenCount = CountTokens(prompt);
        var warnings = new List<string>();

        var compressed = CompressCore(prompt, opts);
        var compressedTokenCount = CountTokens(compressed);

        // Check if we hit minimum token limit
        if (opts.MinTokenCount > 0 && compressedTokenCount < opts.MinTokenCount)
        {
            warnings.Add($"Compression reached minimum token limit ({opts.MinTokenCount})");
        }

        // Check if we achieved target reduction
        var actualReduction = (double)(originalTokenCount - compressedTokenCount) / originalTokenCount;
        if (actualReduction < opts.TargetReduction * 0.5)
        {
            warnings.Add($"Achieved only {actualReduction:P0} reduction, target was {opts.TargetReduction:P0}");
        }

        // Estimate cost savings (using GPT-4 pricing as default: $0.03/1K input tokens)
        var tokensSaved = originalTokenCount - compressedTokenCount;
        var costSavings = (decimal)tokensSaved * 0.00003m;

        return new CompressionResult
        {
            OriginalPrompt = prompt,
            CompressedPrompt = compressed,
            OriginalTokenCount = originalTokenCount,
            CompressedTokenCount = compressedTokenCount,
            EstimatedCostSavings = costSavings,
            CompressionMethod = Name,
            Warnings = warnings.AsReadOnly(),
            CompressedAt = DateTime.UtcNow
        };
    }

    /// <summary>
    /// Compresses a prompt asynchronously.
    /// </summary>
    /// <param name="prompt">The prompt string to compress.</param>
    /// <param name="options">Options controlling compression behavior.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>A task that resolves to the compressed prompt string.</returns>
    public virtual Task<string> CompressAsync(
        string prompt,
        CompressionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(Compress(prompt, options));
    }

    /// <summary>
    /// Core compression logic to be implemented by derived classes.
    /// </summary>
    /// <param name="prompt">The validated prompt to compress.</param>
    /// <param name="options">The compression options.</param>
    /// <returns>The compressed prompt.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> This is where you implement your specific compression algorithm.
    ///
    /// You don't need to:
    /// - Validate the prompt (already done)
    /// - Count tokens (use CountTokens helper if needed)
    /// - Handle null inputs (already validated)
    ///
    /// Just focus on: Making the prompt shorter while preserving meaning.
    /// </para>
    /// </remarks>
    protected abstract string CompressCore(string prompt, CompressionOptions options);

    /// <summary>
    /// Counts tokens in the given text.
    /// </summary>
    /// <param name="text">The text to count tokens in.</param>
    /// <returns>The estimated token count.</returns>
    /// <remarks>
    /// Uses custom token counter if provided, otherwise uses a simple word-based approximation
    /// (1 token ≈ 0.75 words for English).
    /// </remarks>
    protected int CountTokens(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return 0;
        }

        if (_tokenCounter != null)
        {
            return _tokenCounter(text);
        }

        // Simple approximation: split on whitespace and punctuation
        // GPT models average ~0.75 tokens per word for English
        var words = text.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
        return (int)(words.Length / 0.75);
    }

    /// <summary>
    /// Validates the prompt input.
    /// </summary>
    /// <param name="prompt">The prompt to validate.</param>
    protected virtual void ValidatePrompt(string prompt)
    {
        if (prompt == null)
        {
            throw new ArgumentNullException(nameof(prompt));
        }
    }

    /// <summary>
    /// Extracts and preserves variables from a prompt (like {variable_name}).
    /// </summary>
    /// <param name="prompt">The prompt to extract variables from.</param>
    /// <returns>A dictionary mapping placeholder tokens to original variables.</returns>
    protected Dictionary<string, string> ExtractVariables(string prompt)
    {
        var variables = new Dictionary<string, string>();
        var pattern = @"\{[^}]+\}";
        var matches = RegexHelper.Matches(prompt, pattern, RegexOptions.None);

        for (int i = 0; i < matches.Count; i++)
        {
            var placeholder = $"__VAR_{i}__";
            variables[placeholder] = matches[i].Value;
        }

        return variables;
    }

    /// <summary>
    /// Replaces variables with placeholders to protect them during compression.
    /// </summary>
    /// <param name="prompt">The prompt to process.</param>
    /// <param name="variables">The variable mappings.</param>
    /// <returns>The prompt with variables replaced by placeholders.</returns>
    protected string ReplaceVariablesWithPlaceholders(string prompt, Dictionary<string, string> variables)
    {
        var result = prompt;
        foreach (var kvp in variables)
        {
            result = result.Replace(kvp.Value, kvp.Key);
        }
        return result;
    }

    /// <summary>
    /// Restores variables from placeholders after compression.
    /// </summary>
    /// <param name="prompt">The compressed prompt with placeholders.</param>
    /// <param name="variables">The variable mappings.</param>
    /// <returns>The prompt with variables restored.</returns>
    protected string RestoreVariables(string prompt, Dictionary<string, string> variables)
    {
        var result = prompt;
        foreach (var kvp in variables)
        {
            result = result.Replace(kvp.Key, kvp.Value);
        }
        return result;
    }

    /// <summary>
    /// Extracts and preserves code blocks from a prompt.
    /// </summary>
    /// <param name="prompt">The prompt to extract code blocks from.</param>
    /// <returns>A dictionary mapping placeholder tokens to original code blocks.</returns>
    protected Dictionary<string, string> ExtractCodeBlocks(string prompt)
    {
        var codeBlocks = new Dictionary<string, string>();
        var pattern = @"```[\s\S]*?```";
        var matches = RegexHelper.Matches(prompt, pattern, RegexOptions.None);

        for (int i = 0; i < matches.Count; i++)
        {
            var placeholder = $"__CODE_{i}__";
            codeBlocks[placeholder] = matches[i].Value;
        }

        return codeBlocks;
    }

    /// <summary>
    /// Replaces code blocks with placeholders to protect them during compression.
    /// </summary>
    protected string ReplaceCodeBlocksWithPlaceholders(string prompt, Dictionary<string, string> codeBlocks)
    {
        var result = prompt;
        foreach (var kvp in codeBlocks)
        {
            result = result.Replace(kvp.Value, kvp.Key);
        }
        return result;
    }

    /// <summary>
    /// Restores code blocks from placeholders after compression.
    /// </summary>
    protected string RestoreCodeBlocks(string prompt, Dictionary<string, string> codeBlocks)
    {
        var result = prompt;
        foreach (var kvp in codeBlocks)
        {
            result = result.Replace(kvp.Key, kvp.Value);
        }
        return result;
    }
}



