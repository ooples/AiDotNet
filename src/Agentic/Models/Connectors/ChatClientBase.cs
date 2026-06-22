using System.Net.Http;
using AiDotNet.Agentic.Models;
using Newtonsoft.Json;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage, which is in scope
// project-wide via a global using in AiModelBuilder.cs. The agentic subsystem uses the Models type.
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>
/// Base class for <see cref="IChatClient{T}"/> implementations that talk to an HTTP chat API. Provides
/// the shared HTTP client, validation, retry-with-backoff for non-streaming calls, and API-key checks,
/// leaving the provider-specific request/response mapping to subclasses.
/// </summary>
/// <typeparam name="T">The numeric type used across the AiDotNet ecosystem.</typeparam>
/// <remarks>
/// <para>
/// Template-method pattern: this base owns the cross-cutting concerns (transport, retries, timeouts,
/// error wrapping); a concrete connector implements <see cref="GetResponseCoreAsync"/> and
/// <see cref="GetStreamingResponseCoreAsync"/> with the provider's wire format. Streaming calls are not
/// retried (a partially-consumed stream cannot be safely replayed).
/// </para>
/// <para><b>For Beginners:</b> Every cloud chat provider needs the same plumbing — send an HTTP request,
/// retry if the network hiccups, give up cleanly on a bad API key. This class does all of that once, so
/// each provider class only has to describe how its specific API formats requests and responses.
/// </para>
/// </remarks>
public abstract class ChatClientBase<T> : IChatClient<T>
{
    /// <summary>
    /// The HTTP client used for API communication.
    /// </summary>
    protected HttpClient HttpClient { get; }

    /// <summary>
    /// The maximum number of retry attempts for failed non-streaming requests.
    /// </summary>
    protected int MaxRetries { get; set; } = 3;

    /// <summary>
    /// The initial retry delay in milliseconds (doubles with each retry).
    /// </summary>
    protected int InitialRetryDelayMs { get; set; } = 1000;

    /// <summary>
    /// The request timeout in milliseconds.
    /// </summary>
    protected int TimeoutMs { get; set; } = 120000;

    /// <inheritdoc/>
    public string ModelId { get; protected set; } = "unknown";

    /// <summary>
    /// Initializes the base client.
    /// </summary>
    /// <param name="httpClient">An optional HTTP client; a new one is created when <c>null</c>.</param>
    protected ChatClientBase(HttpClient? httpClient)
    {
        if (httpClient is null)
        {
            // We own this instance, so configuring its timeout is safe.
            HttpClient = new HttpClient { Timeout = TimeSpan.FromMilliseconds(TimeoutMs) };
        }
        else
        {
            // Never mutate a caller-owned HttpClient (its Timeout may be shared/intentional); rely on the
            // per-call CancellationToken for time-bounding instead.
            HttpClient = httpClient;
        }
    }

    /// <inheritdoc/>
    public async Task<ChatResponse> GetResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ValidateMessages(messages);
        var effectiveOptions = options ?? new ChatOptions();

        int retryCount = 0;
        int delayMs = InitialRetryDelayMs;
        Exception? lastException = null;

        while (retryCount <= MaxRetries)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                return await GetResponseCoreAsync(messages, effectiveOptions, cancellationToken).ConfigureAwait(false);
            }
            catch (HttpRequestException ex) when (IsRetryable(ex) && retryCount < MaxRetries)
            {
                lastException = ex;
                retryCount++;
                await Task.Delay(delayMs, cancellationToken).ConfigureAwait(false);
                delayMs *= 2;
            }
            catch (TaskCanceledException ex) when (!cancellationToken.IsCancellationRequested && retryCount < MaxRetries)
            {
                // Timeout (not caller cancellation) — retry.
                lastException = ex;
                retryCount++;
                await Task.Delay(delayMs, cancellationToken).ConfigureAwait(false);
                delayMs *= 2;
            }
            catch (JsonException ex)
            {
                throw new InvalidOperationException(
                    $"Failed to parse the response from '{ModelId}'. This usually indicates an API format change or an invalid response.",
                    ex);
            }
        }

        throw new InvalidOperationException(
            $"Failed to get a response from '{ModelId}' after {MaxRetries} retries. Last error: {lastException?.Message}",
            lastException);
    }

    /// <inheritdoc/>
    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ValidateMessages(messages);
        return GetStreamingResponseCoreAsync(messages, options ?? new ChatOptions(), cancellationToken);
    }

    /// <summary>
    /// Provider-specific non-streaming request/response mapping.
    /// </summary>
    /// <param name="messages">The validated, non-empty conversation.</param>
    /// <param name="options">The effective (non-null) options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The complete response.</returns>
    protected abstract Task<ChatResponse> GetResponseCoreAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions options,
        CancellationToken cancellationToken);

    /// <summary>
    /// Provider-specific streaming request/response mapping.
    /// </summary>
    /// <param name="messages">The validated, non-empty conversation.</param>
    /// <param name="options">The effective (non-null) options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The stream of incremental updates.</returns>
    protected abstract IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseCoreAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions options,
        CancellationToken cancellationToken);

    /// <summary>
    /// Determines whether a failed HTTP request should be retried (network errors, 429, 408, 5xx).
    /// </summary>
    /// <param name="ex">The exception to classify.</param>
    /// <returns><c>true</c> when the error is likely transient.</returns>
    protected virtual bool IsRetryable(HttpRequestException ex)
    {
        // Connectors throw HttpResponseException on a non-success response, preserving the status code on
        // every target framework (HttpRequestException.StatusCode does not exist on .NET Framework). Classify
        // from it directly so retry behavior is identical across frameworks.
        if (ex is HttpResponseException responseEx)
        {
            var code = (int)responseEx.ResponseStatusCode;
            return code == 429 || code == 408 || code >= 500;
        }

#if NET5_0_OR_GREATER
        if (ex.StatusCode is null)
        {
            // No HTTP response (transport/network failure) — transient, retry.
            return true;
        }

        var statusCode = (int)ex.StatusCode;
        return statusCode == 429 || statusCode == 408 || statusCode >= 500;
#else
        // .NET Framework: HttpRequestException exposes no StatusCode. Inspect the inner WebException for a
        // network-level failure (timeout/connect/DNS — transient) or an HTTP response, retrying only on
        // 408, 429, or 5xx and treating other HTTP statuses (e.g. 4xx) as permanent.
        for (Exception? inner = ex.InnerException; inner is not null; inner = inner.InnerException)
        {
            if (inner is System.Net.WebException webEx)
            {
                if (webEx.Response is System.Net.HttpWebResponse response)
                {
                    var statusCode = (int)response.StatusCode;
                    return statusCode == 429 || statusCode == 408 || statusCode >= 500;
                }

                switch (webEx.Status)
                {
                    case System.Net.WebExceptionStatus.Timeout:
                    case System.Net.WebExceptionStatus.ConnectFailure:
                    case System.Net.WebExceptionStatus.ConnectionClosed:
                    case System.Net.WebExceptionStatus.ReceiveFailure:
                    case System.Net.WebExceptionStatus.SendFailure:
                    case System.Net.WebExceptionStatus.KeepAliveFailure:
                    case System.Net.WebExceptionStatus.NameResolutionFailure:
                        return true;
                    default:
                        return false;
                }
            }
        }

        // No classifiable inner error: an HttpRequestException with no HTTP response is a transport failure,
        // which is transient.
        return true;
#endif
    }

    /// <summary>
    /// Validates that an API key is present.
    /// </summary>
    /// <param name="apiKey">The API key.</param>
    /// <param name="paramName">The parameter name for error messages.</param>
    /// <exception cref="ArgumentNullException">Thrown when the key is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when the key is empty/whitespace.</exception>
    protected static void ValidateApiKey(string? apiKey, string paramName = "apiKey")
    {
        if (apiKey is null)
        {
            throw new ArgumentNullException(paramName, "API key cannot be null.");
        }

        if (string.IsNullOrWhiteSpace(apiKey))
        {
            throw new ArgumentException("API key cannot be empty or whitespace.", paramName);
        }
    }

    private static void ValidateMessages(IReadOnlyList<ChatMessage> messages)
    {
        Guard.NotNull(messages);
        if (messages.Count == 0)
        {
            throw new ArgumentException("At least one message is required.", nameof(messages));
        }
    }
}
