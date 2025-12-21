using System.Net.Http;
using AiDotNet.Interfaces;
using Newtonsoft.Json;

namespace AiDotNet.LanguageModels;

/// <summary>
/// Provides a base implementation for chat-based language models with common functionality
/// including HTTP communication, retry logic, rate limiting, and error handling.
/// </summary>
/// <typeparam name="T">The numeric type used for model parameters and operations (e.g., double, float).</typeparam>
/// <remarks>
/// For Beginners:
/// This base class handles all the complex infrastructure needed to communicate with language model APIs:
/// - Making HTTP requests to API endpoints
/// - Retrying failed requests (network issues, temporary errors)
/// - Handling rate limits (not sending too many requests too fast)
/// - Managing timeouts
/// - Logging errors
///
/// When creating a new language model implementation (like OpenAIChatModel or AnthropicChatModel):
/// 1. Inherit from this class
/// 2. Set ModelName, MaxContextTokens, MaxGenerationTokens in your constructor
/// 3. Implement GenerateAsyncCore() with your specific API logic
/// 4. Everything else (retries, rate limiting, error handling) is handled automatically
///
/// This design pattern is called the "Template Method Pattern" - the base class provides
/// the structure and common code, while derived classes fill in the specific details.
/// </remarks>
public abstract class ChatModelBase<T> : IChatModel<T>
{
    /// <summary>
    /// The HTTP client used for API communication.
    /// </summary>
    protected readonly HttpClient HttpClient;

    /// <summary>
    /// The maximum number of retry attempts for failed requests.
    /// </summary>
    protected int MaxRetries { get; set; } = 3;

    /// <summary>
    /// The initial retry delay in milliseconds (doubles with each retry).
    /// </summary>
    protected int InitialRetryDelayMs { get; set; } = 1000;

    /// <summary>
    /// The request timeout in milliseconds.
    /// </summary>
    protected int TimeoutMs { get; set; } = 120000; // 2 minutes

    /// <summary>
    /// Gets or sets whether to log detailed error information.
    /// </summary>
    protected bool EnableDetailedLogging { get; set; } = false;

    /// <inheritdoc/>
    public string ModelName { get; protected set; } = "unknown";

    /// <inheritdoc/>
    public int MaxContextTokens { get; protected set; }

    /// <inheritdoc/>
    public int MaxGenerationTokens { get; protected set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ChatModelBase{T}"/> class.
    /// </summary>
    /// <param name="httpClient">Optional HTTP client. If null, a new one will be created.</param>
    /// <param name="maxContextTokens">The maximum context window size in tokens.</param>
    /// <param name="maxGenerationTokens">The maximum number of tokens to generate.</param>
    /// <remarks>
    /// For Beginners:
    /// This constructor sets up the infrastructure for communicating with language model APIs.
    ///
    /// The httpClient parameter:
    /// - If you pass null, a new HttpClient is created automatically
    /// - If you pass your own, you can customize headers, timeouts, etc.
    /// - In production, consider using IHttpClientFactory for better performance
    ///
    /// The token parameters control how much text the model can process and generate:
    /// - maxContextTokens: How much text can be in the prompt
    /// - maxGenerationTokens: How long the response can be
    /// </remarks>
    protected ChatModelBase(HttpClient? httpClient, int maxContextTokens, int maxGenerationTokens)
    {
        if (maxContextTokens <= 0)
        {
            throw new ArgumentException("Maximum context tokens must be positive.", nameof(maxContextTokens));
        }

        if (maxGenerationTokens <= 0)
        {
            throw new ArgumentException("Maximum generation tokens must be positive.", nameof(maxGenerationTokens));
        }

        HttpClient = httpClient ?? new HttpClient();
        HttpClient.Timeout = TimeSpan.FromMilliseconds(TimeoutMs);
        MaxContextTokens = maxContextTokens;
        MaxGenerationTokens = maxGenerationTokens;
    }

    /// <inheritdoc/>
    public async Task<string> GenerateAsync(string prompt, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(prompt))
        {
            throw new ArgumentException("Prompt cannot be null or whitespace.", nameof(prompt));
        }

        // Check for cancellation before starting
        cancellationToken.ThrowIfCancellationRequested();

        // Estimate token count and warn if it might exceed limits
        int estimatedTokens = EstimateTokenCount(prompt);
        if (estimatedTokens > MaxContextTokens)
        {
            throw new ArgumentException(
                $"Prompt is too long ({estimatedTokens} estimated tokens exceeds {MaxContextTokens} max context tokens). " +
                $"Please reduce the prompt length.",
                nameof(prompt));
        }

        // Retry loop with exponential backoff
        int retryCount = 0;
        int delayMs = InitialRetryDelayMs;
        Exception? lastException = null;

        while (retryCount <= MaxRetries)
        {
            try
            {
                return await GenerateAsyncCore(prompt, cancellationToken);
            }
            catch (HttpRequestException ex) when (IsRetryable(ex) && retryCount < MaxRetries)
            {
                lastException = ex;
                retryCount++;

                if (EnableDetailedLogging)
                {
                    Console.WriteLine($"Request failed (attempt {retryCount}/{MaxRetries}): {ex.Message}");
                    Console.WriteLine($"Retrying in {delayMs}ms...");
                }

                await Task.Delay(delayMs, cancellationToken);
                delayMs *= 2; // Exponential backoff
            }
            catch (TaskCanceledException ex) when (retryCount < MaxRetries)
            {
                // Timeout - retry
                lastException = ex;
                retryCount++;

                if (EnableDetailedLogging)
                {
                    Console.WriteLine($"Request timed out (attempt {retryCount}/{MaxRetries})");
                    Console.WriteLine($"Retrying in {delayMs}ms...");
                }

                await Task.Delay(delayMs, cancellationToken);
                delayMs *= 2;
            }
            catch (JsonException ex)
            {
                // JSON parsing error - don't retry
                throw new InvalidOperationException(
                    $"Failed to parse API response from {ModelName}. " +
                    $"This usually indicates an API format change or invalid response.",
                    ex);
            }
            catch (Exception ex)
            {
                // Other errors - wrap and throw
                throw new InvalidOperationException(
                    $"Error generating response from {ModelName}: {ex.Message}",
                    ex);
            }
        }

        // All retries exhausted
        throw new InvalidOperationException(
            $"Failed to generate response from {ModelName} after {MaxRetries} retries. " +
            $"Last error: {lastException?.Message}",
            lastException);
    }

    /// <inheritdoc/>
    public string Generate(string prompt)
    {
        // Synchronous wrapper around async method
        // Note: In production, prefer using GenerateAsync when possible
        return GenerateAsync(prompt).GetAwaiter().GetResult();
    }

    /// <inheritdoc/>
    public Task<string> GenerateResponseAsync(string prompt, CancellationToken cancellationToken = default)
    {
        // Alias for GenerateAsync for IChatModel compatibility
        // Now properly propagates cancellation token
        return GenerateAsync(prompt, cancellationToken);
    }

    /// <summary>
    /// Core generation logic to be implemented by derived classes.
    /// This method should make the actual API call to the language model.
    /// </summary>
    /// <param name="prompt">The validated prompt string.</param>
    /// <param name="cancellationToken">Cancellation token to cancel the operation.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains the generated response.</returns>
    /// <remarks>
    /// For Beginners:
    /// This is where you implement the specific logic for your language model API.
    /// The base class handles retries, timeouts, and error handling, so you only need to:
    /// 1. Build the API request (JSON body, headers, etc.)
    /// 2. Send it to the API endpoint
    /// 3. Parse the response
    /// 4. Return the generated text
    ///
    /// If anything goes wrong, just throw an exception - the base class will handle retries.
    ///
    /// IMPORTANT: Pass the cancellationToken to your HTTP calls (e.g., HttpClient.PostAsync)
    /// to enable proper cancellation support.
    /// </remarks>
    protected abstract Task<string> GenerateAsyncCore(string prompt, CancellationToken cancellationToken);

    /// <summary>
    /// Estimates the number of tokens in a text string.
    /// This is a rough approximation - actual tokenization varies by model.
    /// </summary>
    /// <param name="text">The text to estimate tokens for.</param>
    /// <returns>The estimated number of tokens.</returns>
    /// <remarks>
    /// For Beginners:
    /// Tokens are the units that language models process. They're not quite words:
    /// - Common words are usually 1 token: "cat", "the", "is"
    /// - Uncommon words might be 2-3 tokens: "ChatGPT" = "Chat" + "G" + "PT"
    /// - Numbers and punctuation: varies
    ///
    /// This method uses a simple heuristic: 1 token ≈ 4 characters (or 0.75 words).
    /// This works reasonably well for English but is less accurate for other languages.
    ///
    /// For precise token counts, use the model's official tokenizer (e.g., tiktoken for OpenAI).
    /// </remarks>
    protected virtual int EstimateTokenCount(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return 0;
        }

        // Rough estimate: 1 token ≈ 4 characters
        // This is approximate and varies by model and language
        return (int)Math.Ceiling(text.Length / 4.0);
    }

    /// <summary>
    /// Determines whether an exception is retryable.
    /// </summary>
    /// <param name="ex">The exception to check.</param>
    /// <returns>True if the exception indicates a transient error that might succeed on retry.</returns>
    /// <remarks>
    /// For Beginners:
    /// Some errors are temporary and worth retrying:
    /// - Network timeouts
    /// - Server overload (503 errors)
    /// - Rate limiting (429 errors)
    ///
    /// Other errors are permanent and won't be fixed by retrying:
    /// - Invalid API key (401 errors)
    /// - Malformed request (400 errors)
    /// - Resource not found (404 errors)
    ///
    /// This method helps distinguish between the two.
    /// </remarks>
    protected virtual bool IsRetryable(HttpRequestException ex)
    {
#if NET5_0_OR_GREATER
        // Retry on network errors or server errors (5xx)
        if (ex.StatusCode == null)
        {
            // Network error (no status code)
            return true;
        }

        var statusCode = (int)ex.StatusCode;

        // Retry on:
        // - 429 (Rate Limit)
        // - 500+ (Server Errors)
        // - 408 (Request Timeout)
        return statusCode == 429 || statusCode >= 500 || statusCode == 408;
#else
        // For .NET Framework, assume network errors are retryable
        // since HttpRequestException doesn't expose StatusCode
        return true;
#endif
    }

    /// <summary>
    /// Validates the API key is not empty.
    /// </summary>
    /// <param name="apiKey">The API key to validate.</param>
    /// <param name="paramName">The parameter name for error messages.</param>
    /// <exception cref="ArgumentNullException">Thrown when the API key is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the API key is empty or whitespace.</exception>
    /// <remarks>
    /// For Beginners:
    /// Most language model APIs require an API key for authentication. This is like a password
    /// that proves you're authorized to use the service and allows billing.
    ///
    /// This helper method checks that:
    /// - The API key exists (not null)
    /// - The API key isn't just whitespace
    ///
    /// Always keep API keys secret! Don't commit them to git or share them publicly.
    /// </remarks>
    protected static void ValidateApiKey(string? apiKey, string paramName = "apiKey")
    {
        if (apiKey == null)
        {
            throw new ArgumentNullException(paramName, "API key cannot be null.");
        }

        if (string.IsNullOrWhiteSpace(apiKey))
        {
            throw new ArgumentException("API key cannot be empty or whitespace.", paramName);
        }
    }
}
