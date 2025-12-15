namespace AiDotNet.PromptEngineering;

/// <summary>
/// Manages context window limits for LLM prompts, providing token estimation and text truncation utilities.
/// </summary>
/// <remarks>
/// <para>
/// This class helps manage the token limits of large language models by providing utilities
/// to estimate token counts, check if text fits within the context window, and truncate
/// or split text that exceeds the limit.
/// </para>
/// <para><b>For Beginners:</b> Large Language Models have a maximum number of tokens they can process
/// at once (the "context window"). This class helps you:
///
/// Example:
/// ```csharp
/// // Create a manager with 4096 token limit
/// var manager = new ContextWindowManager(4096);
///
/// // Check if your prompt fits
/// var prompt = "Your long prompt here...";
/// if (!manager.FitsInWindow(prompt))
/// {
///     // Truncate to fit
///     prompt = manager.TruncateToFit(prompt);
/// }
///
/// // Or split long text into chunks
/// var chunks = manager.SplitIntoChunks(longDocument);
/// ```
///
/// A token is roughly 4 characters or 0.75 words in English, but varies by language and model.
/// </para>
/// </remarks>
public class ContextWindowManager
{
    private readonly Func<string, int> _tokenEstimator;

    /// <summary>
    /// Gets the maximum number of tokens allowed in the context window.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the total capacity of the model's context window.
    /// Common values are 4096, 8192, 16384, 32768, or 128000 depending on the model.
    /// </para>
    /// </remarks>
    public int MaxTokens { get; }

    /// <summary>
    /// Initializes a new instance of the ContextWindowManager with the specified maximum tokens.
    /// </summary>
    /// <param name="maxTokens">The maximum number of tokens allowed in the context window.</param>
    /// <remarks>
    /// <para>
    /// Uses a default token estimator that approximates tokens as text.Length / 4,
    /// which is a reasonable approximation for English text.
    /// </para>
    /// <para><b>For Beginners:</b> Just specify the token limit for your model:
    ///
    /// ```csharp
    /// // For GPT-4 with 8K context
    /// var manager = new ContextWindowManager(8192);
    ///
    /// // For Claude with 100K context
    /// var manager = new ContextWindowManager(100000);
    /// ```
    /// </para>
    /// </remarks>
    public ContextWindowManager(int maxTokens)
        : this(maxTokens, DefaultTokenEstimator)
    {
    }

    /// <summary>
    /// Initializes a new instance of the ContextWindowManager with a custom token estimator.
    /// </summary>
    /// <param name="maxTokens">The maximum number of tokens allowed in the context window.</param>
    /// <param name="tokenEstimator">A function that estimates the number of tokens in a given text.</param>
    /// <remarks>
    /// <para>
    /// Use this constructor when you need precise token counting, such as when using
    /// a specific tokenizer like tiktoken for OpenAI models.
    /// </para>
    /// <para><b>For Beginners:</b> Use a custom estimator for more accurate counting:
    ///
    /// ```csharp
    /// // Simple character-based estimator
    /// var manager = new ContextWindowManager(4096, text => text.Length / 4);
    ///
    /// // Word-based estimator
    /// var manager = new ContextWindowManager(4096, text =>
    ///     text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length * 4 / 3);
    /// ```
    /// </para>
    /// </remarks>
    public ContextWindowManager(int maxTokens, Func<string, int> tokenEstimator)
    {
        MaxTokens = maxTokens;
        _tokenEstimator = tokenEstimator ?? DefaultTokenEstimator;
    }

    /// <summary>
    /// Estimates the number of tokens in the given text.
    /// </summary>
    /// <param name="text">The text to estimate tokens for.</param>
    /// <returns>The estimated number of tokens.</returns>
    /// <remarks>
    /// <para>
    /// The accuracy depends on the token estimator provided. The default estimator
    /// uses a simple character-based approximation.
    /// </para>
    /// <para><b>For Beginners:</b> Use this to check how many tokens your text uses:
    ///
    /// ```csharp
    /// var tokens = manager.EstimateTokens("Hello, world!");
    /// Console.WriteLine($"This text uses approximately {tokens} tokens");
    /// ```
    /// </para>
    /// </remarks>
    public int EstimateTokens(string text)
    {
        if (string.IsNullOrEmpty(text))
            return 0;

        return _tokenEstimator(text);
    }

    /// <summary>
    /// Checks if the given text fits within the context window.
    /// </summary>
    /// <param name="text">The text to check.</param>
    /// <param name="reservedTokens">Number of tokens to reserve for other content (e.g., system prompt, response).</param>
    /// <returns>True if the text fits within the available window space; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// The reserved tokens parameter allows you to account for tokens that will be used
    /// by system prompts, few-shot examples, or expected response length.
    /// </para>
    /// <para><b>For Beginners:</b> Check if your prompt will fit:
    ///
    /// ```csharp
    /// // Basic check
    /// if (manager.FitsInWindow(myPrompt))
    /// {
    ///     // Safe to send to the model
    /// }
    ///
    /// // Reserve space for the response
    /// if (manager.FitsInWindow(myPrompt, reservedTokens: 500))
    /// {
    ///     // Prompt fits with 500 tokens reserved for response
    /// }
    /// ```
    /// </para>
    /// </remarks>
    public bool FitsInWindow(string text, int reservedTokens = 0)
    {
        var estimatedTokens = EstimateTokens(text);
        return estimatedTokens <= MaxTokens - reservedTokens;
    }

    /// <summary>
    /// Calculates the remaining tokens available after accounting for the given text.
    /// </summary>
    /// <param name="text">The text currently using the window.</param>
    /// <param name="reservedTokens">Number of tokens to reserve for other content.</param>
    /// <returns>The number of remaining tokens available, or 0 if exceeded.</returns>
    /// <remarks>
    /// <para>
    /// This is useful for determining how much additional content can be added to a prompt.
    /// </para>
    /// <para><b>For Beginners:</b> Find out how much space is left:
    ///
    /// ```csharp
    /// var remaining = manager.RemainingTokens(currentPrompt);
    /// Console.WriteLine($"You can add approximately {remaining} more tokens");
    /// ```
    /// </para>
    /// </remarks>
    public int RemainingTokens(string text, int reservedTokens = 0)
    {
        var usedTokens = EstimateTokens(text);
        return Math.Max(0, MaxTokens - usedTokens - reservedTokens);
    }

    /// <summary>
    /// Truncates the text to fit within the context window.
    /// </summary>
    /// <param name="text">The text to truncate.</param>
    /// <param name="reservedTokens">Number of tokens to reserve for other content.</param>
    /// <returns>The truncated text that fits within the window, or the original if it already fits.</returns>
    /// <remarks>
    /// <para>
    /// Uses binary search to find the optimal truncation point, ensuring the result
    /// fits within the available token budget.
    /// </para>
    /// <para><b>For Beginners:</b> Automatically shorten text to fit:
    ///
    /// ```csharp
    /// var longDocument = LoadDocument(); // Very long text
    /// var fittingText = manager.TruncateToFit(longDocument);
    /// // fittingText is guaranteed to fit in the context window
    /// ```
    /// </para>
    /// </remarks>
    public string TruncateToFit(string text, int reservedTokens = 0)
    {
        if (string.IsNullOrEmpty(text))
            return text ?? "";

        if (FitsInWindow(text, reservedTokens))
            return text;

        var availableTokens = MaxTokens - reservedTokens;
        if (availableTokens <= 0)
            return "";

        // Binary search for the right length
        int low = 0;
        int high = text.Length;
        int bestFit = 0;

        while (low <= high)
        {
            int mid = (low + high) / 2;
            var candidate = text.Substring(0, mid);

            if (EstimateTokens(candidate) <= availableTokens)
            {
                bestFit = mid;
                low = mid + 1;
            }
            else
            {
                high = mid - 1;
            }
        }

        return text.Substring(0, bestFit);
    }

    /// <summary>
    /// Splits the text into chunks that each fit within the context window.
    /// </summary>
    /// <param name="text">The text to split.</param>
    /// <param name="overlapTokens">Number of tokens to overlap between chunks for context continuity.</param>
    /// <returns>A list of text chunks that each fit within the window.</returns>
    /// <remarks>
    /// <para>
    /// This is useful for processing long documents that exceed the context window.
    /// The overlap parameter allows maintaining context between chunks.
    /// </para>
    /// <para><b>For Beginners:</b> Process long documents in pieces:
    ///
    /// ```csharp
    /// var document = LoadLongDocument();
    /// var chunks = manager.SplitIntoChunks(document);
    ///
    /// foreach (var chunk in chunks)
    /// {
    ///     var response = await model.ProcessAsync(chunk);
    ///     // Handle each chunk's response
    /// }
    ///
    /// // With overlap for better context continuity
    /// var overlappingChunks = manager.SplitIntoChunks(document, overlapTokens: 100);
    /// ```
    /// </para>
    /// </remarks>
    public List<string> SplitIntoChunks(string text, int overlapTokens = 0)
    {
        var chunks = new List<string>();

        if (string.IsNullOrEmpty(text))
            return chunks;

        if (FitsInWindow(text))
        {
            chunks.Add(text);
            return chunks;
        }

        int startIndex = 0;
        while (startIndex < text.Length)
        {
            // Find the largest chunk that fits
            var remainingText = text.Substring(startIndex);
            var chunk = TruncateToFit(remainingText);

            if (string.IsNullOrEmpty(chunk))
            {
                // Can't fit even one character - shouldn't happen with positive MaxTokens
                break;
            }

            chunks.Add(chunk);

            // Calculate overlap in characters (approximate)
            int overlapChars = 0;
            if (overlapTokens > 0 && chunk.Length > 0)
            {
                // Estimate characters per token based on current chunk
                double charsPerToken = (double)chunk.Length / Math.Max(1, EstimateTokens(chunk));
                overlapChars = (int)(overlapTokens * charsPerToken);
                overlapChars = Math.Min(overlapChars, chunk.Length - 1);
            }

            startIndex += chunk.Length - overlapChars;

            // Ensure we always make progress
            if (startIndex <= 0 && chunks.Count > 0)
                break;
        }

        return chunks;
    }

    /// <summary>
    /// Default token estimator that approximates tokens as roughly 1 token per 4 characters.
    /// </summary>
    /// <param name="text">The text to estimate tokens for.</param>
    /// <returns>The estimated token count.</returns>
    /// <remarks>
    /// <para>
    /// This is a simple heuristic that works reasonably well for English text.
    /// For more accurate counting, use a model-specific tokenizer.
    /// </para>
    /// </remarks>
    private static int DefaultTokenEstimator(string text)
    {
        if (string.IsNullOrEmpty(text))
            return 0;

        // Rough approximation: ~4 characters per token on average
        return (int)Math.Ceiling(text.Length / 4.0);
    }
}
