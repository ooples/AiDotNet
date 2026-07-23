using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.PromptEngineering.Compression;

/// <summary>
/// Compressor that chains multiple compressors together in sequence.
/// </summary>
/// <remarks>
/// <para>
/// This compressor applies multiple compression strategies in sequence, with each
/// compressor working on the output of the previous one. This allows combining
/// the strengths of different compression approaches.
/// </para>
/// <para><b>For Beginners:</b> Combines multiple compressors for better results.
///
/// Example:
/// <code>
/// // Create a pipeline of compressors
/// var compressor = new CompositeCompressor(
///     new RedundancyRemovalCompressor(),    // First: remove redundant phrases
///     new SentenceCompressor(),             // Then: simplify sentences
///     new StopWordRemovalCompressor()       // Finally: remove stop words
/// );
///
/// // Apply all compressors in sequence
/// var result = compressor.Compress(longPrompt);
/// // Result: Prompt has been compressed by all three methods
/// </code>
///
/// Order matters:
/// - Put rule-based compressors first
/// - Put aggressive compressors last
/// - Each compressor should work well with the output of the previous one
/// </para>
/// </remarks>
public class CompositeCompressor : PromptCompressorBase
{
    private readonly List<IPromptCompressor> _compressors;

    /// <summary>
    /// Initializes a new instance of the CompositeCompressor class.
    /// </summary>
    /// <param name="compressors">The compressors to apply in sequence.</param>
    public CompositeCompressor(params IPromptCompressor[] compressors)
        : this(compressors.AsEnumerable())
    {
    }

    /// <summary>
    /// Initializes a new instance of the CompositeCompressor class.
    /// </summary>
    /// <param name="compressors">The compressors to apply in sequence.</param>
    /// <param name="tokenCounter">Optional custom token counter function.</param>
    public CompositeCompressor(
        IEnumerable<IPromptCompressor> compressors,
        Func<string, int>? tokenCounter = null)
        : base(BuildName(compressors), tokenCounter)
    {
        Guard.NotNull(compressors);
        _compressors = compressors.ToList();

        if (_compressors.Count == 0)
        {
            throw new ArgumentException("At least one compressor is required.", nameof(compressors));
        }
    }

    /// <summary>
    /// Builds the name of the composite compressor from its components.
    /// </summary>
    private static string BuildName(IEnumerable<IPromptCompressor>? compressors)
    {
        if (compressors == null)
        {
            return "CompositeCompressor";
        }

        var names = compressors.Select(c => c.Name).ToList();
        if (names.Count <= 3)
        {
            return $"Composite({string.Join("+", names)})";
        }

        return $"Composite({names.Count} compressors)";
    }

    /// <summary>
    /// Gets the list of compressors in this composite.
    /// </summary>
    public IReadOnlyList<IPromptCompressor> Compressors => _compressors.AsReadOnly();

    /// <summary>
    /// Compresses the prompt by applying all compressors in sequence.
    /// </summary>
    protected override string CompressCore(string prompt, CompressionOptions options)
    {
        var result = prompt;

        foreach (var compressor in _compressors)
        {
            result = compressor.Compress(result, options);

            // Check if we've achieved target reduction
            var originalTokens = CountTokens(prompt);
            var currentTokens = CountTokens(result);
            var currentReduction = (double)(originalTokens - currentTokens) / originalTokens;

            if (currentReduction >= options.TargetReduction)
            {
                break; // Target achieved, stop early
            }

            // Check minimum token limit
            if (options.MinTokenCount > 0 && currentTokens <= options.MinTokenCount)
            {
                break; // Don't compress further
            }
        }

        return result;
    }

    /// <summary>
    /// Compresses the prompt asynchronously by applying all compressors in sequence.
    /// </summary>
    public override async Task<string> CompressAsync(
        string prompt,
        CompressionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var opts = options ?? CompressionOptions.Default;
        var result = prompt;

        foreach (var compressor in _compressors)
        {
            cancellationToken.ThrowIfCancellationRequested();

            result = await compressor.CompressAsync(result, opts, cancellationToken)
                .ConfigureAwait(false);

            // Check if we've achieved target reduction
            var originalTokens = CountTokens(prompt);
            var currentTokens = CountTokens(result);
            var currentReduction = (double)(originalTokens - currentTokens) / originalTokens;

            if (currentReduction >= opts.TargetReduction)
            {
                break; // Target achieved, stop early
            }

            // Check minimum token limit
            if (opts.MinTokenCount > 0 && currentTokens <= opts.MinTokenCount)
            {
                break; // Don't compress further
            }
        }

        return result;
    }

    /// <summary>
    /// Creates a composite compressor with a standard pipeline for general use.
    /// </summary>
    /// <returns>A composite compressor with redundancy, sentence, and light stop word removal.</returns>
    public static CompositeCompressor CreateStandardPipeline()
    {
        return new CompositeCompressor(
            new RedundancyRemovalCompressor(),
            new SentenceCompressor()
        );
    }

    /// <summary>
    /// Creates a composite compressor with an aggressive pipeline for maximum compression.
    /// </summary>
    /// <returns>A composite compressor with all available compression techniques.</returns>
    public static CompositeCompressor CreateAggressivePipeline()
    {
        return new CompositeCompressor(
            new RedundancyRemovalCompressor(),
            new SentenceCompressor(),
            new StopWordRemovalCompressor(StopWordRemovalCompressor.AggressivenessLevel.Aggressive)
        );
    }
}
