using System.Globalization;
using System.Net.Http;
using System.Net.Http.Headers;
using AiDotNet.Agentic.Models.Connectors;
using AiDotNet.Interfaces;
using AiDotNet.Validation;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Embeddings;

/// <summary>An <see cref="IEmbeddingClient"/> for any provider that speaks the OpenAI <c>/v1/embeddings</c> wire format.</summary>
/// <remarks>
/// <para>
/// The transport mirrors <see cref="AiDotNet.Agentic.Models.Connectors.ChatClientBase{T}"/> exactly: a caller-owned
/// <see cref="System.Net.Http.HttpClient"/> is never mutated, an owned one gets the configured timeout, a
/// non-success response becomes an <see cref="HttpResponseException"/> so the status survives on .NET Framework,
/// and retries use the same doubling backoff over 429, 408, 5xx and transport errors — with the same
/// <c>NET5_0_OR_GREATER</c> split, because <c>HttpRequestException.StatusCode</c> does not exist on net471 and the
/// inner <see cref="System.Net.WebException"/> has to be inspected instead.
/// </para>
/// <para>
/// Three things differ from the reference implementation deliberately. Upstream accepts only a hard-coded list of
/// model names and picks a provider from that name, so an OpenAI-compatible gateway, a local server, or any new
/// model is a <c>ValueError</c>; here the endpoint is a parameter and any model name is legal. Upstream has no
/// retry and no timeout at all. And upstream returns an empty vector on failure, which its caller then stores as
/// that program's embedding forever; here a failure is an explicit <see cref="EmbeddingBatch.Failure"/> carrying
/// only a status code or exception type — never the response body, which can echo request content.
/// </para>
/// <para>
/// Each input is truncated to <see cref="MaxInputCharacters"/> before it is sent, because the text being embedded is
/// usually a model-authored program and an unbounded candidate must not become an unbounded request body. The API
/// key is held in memory for the lifetime of the client and is never logged, serialized, or persisted.
/// </para>
/// <para><b>For Beginners:</b> This is the adapter that asks a real provider to turn text into vectors. You give it
/// your API key and, if you are not using OpenAI itself, the URL of a compatible service. It handles the network
/// details — timeouts, retrying a request that failed for a temporary reason, and reporting a permanent failure in
/// a way that cannot be mistaken for a real answer.</para>
/// </remarks>
public sealed class OpenAICompatibleEmbeddingClient : IEmbeddingClient
{
    /// <summary>The public OpenAI embeddings endpoint used when none is supplied.</summary>
    public const string DefaultEndpoint = "https://api.openai.com/v1/embeddings";

    /// <summary>The model requested when none is supplied.</summary>
    public const string DefaultModelId = "text-embedding-3-small";

    /// <summary>The default per-input character bound.</summary>
    public const int DefaultMaxInputCharacters = 24_000;

    private static readonly JsonSerializerSettings JsonSettings =
        new() { NullValueHandling = NullValueHandling.Ignore };

    private readonly string _apiKey;
    private readonly int _maxRetries;
    private readonly int _initialRetryDelayMilliseconds;

    /// <summary>Initializes an embeddings client.</summary>
    /// <param name="apiKey">The provider API key.</param>
    /// <param name="modelId">The embedding model name; defaults to <see cref="DefaultModelId"/>.</param>
    /// <param name="endpoint">The embeddings URL; defaults to <see cref="DefaultEndpoint"/>.</param>
    /// <param name="httpClient">An optional caller-owned HTTP client; a new one is created when <c>null</c>.</param>
    /// <param name="maxInputCharacters">The per-input character bound; defaults to <see cref="DefaultMaxInputCharacters"/>.</param>
    /// <param name="maxRetries">How many times a transient failure is retried; 0 to 8.</param>
    /// <param name="initialRetryDelayMilliseconds">The first backoff delay, which doubles per retry.</param>
    /// <param name="timeoutMilliseconds">The request timeout applied to an owned HTTP client.</param>
    /// <exception cref="ArgumentNullException"><paramref name="apiKey"/>, <paramref name="modelId"/>, or <paramref name="endpoint"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">A string argument is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A numeric argument is outside its permitted range.</exception>
    public OpenAICompatibleEmbeddingClient(
        string apiKey,
        string modelId = DefaultModelId,
        string? endpoint = null,
        HttpClient? httpClient = null,
        int maxInputCharacters = DefaultMaxInputCharacters,
        int maxRetries = 3,
        int initialRetryDelayMilliseconds = 1000,
        int timeoutMilliseconds = 120_000)
    {
        Guard.NotNullOrWhiteSpace(apiKey);
        Guard.NotNullOrWhiteSpace(modelId);
        if (maxInputCharacters <= 0 || maxInputCharacters > 1_000_000)
        {
            throw new ArgumentOutOfRangeException(nameof(maxInputCharacters), maxInputCharacters,
                "Value must be between 1 and 1000000.");
        }

        if (maxRetries < 0 || maxRetries > 8)
        {
            throw new ArgumentOutOfRangeException(nameof(maxRetries), maxRetries, "Value must be between 0 and 8.");
        }

        if (initialRetryDelayMilliseconds < 0 || initialRetryDelayMilliseconds > 60_000)
        {
            throw new ArgumentOutOfRangeException(nameof(initialRetryDelayMilliseconds),
                initialRetryDelayMilliseconds, "Value must be between 0 and 60000.");
        }

        if (timeoutMilliseconds <= 0 || timeoutMilliseconds > 600_000)
        {
            throw new ArgumentOutOfRangeException(nameof(timeoutMilliseconds), timeoutMilliseconds,
                "Value must be between 1 and 600000.");
        }

        string resolvedEndpoint = endpoint ?? DefaultEndpoint;
        Guard.NotNullOrWhiteSpace(resolvedEndpoint, nameof(endpoint));

        _apiKey = apiKey;
        _maxRetries = maxRetries;
        _initialRetryDelayMilliseconds = initialRetryDelayMilliseconds;
        MaxInputCharacters = maxInputCharacters;
        Endpoint = resolvedEndpoint;
        ModelId = modelId;

        // Never mutate a caller-owned HttpClient; its timeout may be shared and intentional.
        HttpClient = httpClient ?? new HttpClient { Timeout = TimeSpan.FromMilliseconds(timeoutMilliseconds) };
    }

    /// <inheritdoc/>
    public string ModelId { get; }

    /// <summary>Gets the endpoint requests are posted to.</summary>
    public string Endpoint { get; }

    /// <summary>Gets the per-input character bound applied before a request is sent.</summary>
    public int MaxInputCharacters { get; }

    /// <summary>Gets the HTTP client used for provider communication.</summary>
    private HttpClient HttpClient { get; }

    /// <inheritdoc/>
    public async ValueTask<EmbeddingBatch> EmbedAsync(
        IReadOnlyList<string> texts,
        CancellationToken cancellationToken = default)
    {
        EmbeddingRequestValidation.Validate(texts);

        var payload = new JObject
        {
            ["model"] = ModelId,
            ["input"] = BuildInput(texts),
            ["encoding_format"] = "float"
        };

        int retryCount = 0;
        int delayMilliseconds = _initialRetryDelayMilliseconds;
        string lastProblem = "no reason was recorded";

        while (retryCount <= _maxRetries)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                return await SendAsync(payload, texts.Count, cancellationToken).ConfigureAwait(false);
            }
            catch (HttpRequestException exception) when (IsRetryable(exception) && retryCount < _maxRetries)
            {
                lastProblem = DescribeProblem(exception);
                retryCount++;
                await Task.Delay(delayMilliseconds, cancellationToken).ConfigureAwait(false);
                delayMilliseconds *= 2;
            }
            catch (HttpRequestException exception)
            {
                return EmbeddingBatch.Failure(DescribeProblem(exception));
            }
            catch (TaskCanceledException exception)
                when (!cancellationToken.IsCancellationRequested && retryCount < _maxRetries)
            {
                // A timeout rather than caller cancellation: retry, exactly as the chat transport does.
                lastProblem = "the request timed out (" + exception.GetType().Name + ")";
                retryCount++;
                await Task.Delay(delayMilliseconds, cancellationToken).ConfigureAwait(false);
                delayMilliseconds *= 2;
            }
            catch (TaskCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            catch (TaskCanceledException exception)
            {
                return EmbeddingBatch.Failure("the request timed out (" + exception.GetType().Name + ")");
            }
            catch (JsonException exception)
            {
                // A malformed body is not transient; the provider's contract changed or a proxy rewrote it.
                return EmbeddingBatch.Failure("the response was not valid JSON (" + exception.GetType().Name + ")");
            }
        }

        return EmbeddingBatch.Failure(
            "gave up after " + _maxRetries.ToString(CultureInfo.InvariantCulture) + " retries: " + lastProblem);
    }

    private async Task<EmbeddingBatch> SendAsync(JObject payload, int expectedCount, CancellationToken cancellationToken)
    {
        string json = JsonConvert.SerializeObject(payload, JsonSettings);
        using var request = new HttpRequestMessage(HttpMethod.Post, Endpoint)
        {
            Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json")
        };
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);

        using HttpResponseMessage response = await HttpClient
            .SendAsync(request, cancellationToken)
            .ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            // The body is deliberately not read into the message: it can echo the request, and the request holds
            // program text. The status code is all the retry classifier and the caller need.
            throw new HttpResponseException(
                response.StatusCode,
                "The embeddings request failed with status " +
                ((int)response.StatusCode).ToString(CultureInfo.InvariantCulture) + ".");
        }

        string body = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
        return ReadVectors(body, expectedCount);
    }

    private static EmbeddingBatch ReadVectors(string body, int expectedCount)
    {
        var root = JObject.Parse(body);
        if (root["data"] is not JArray data || data.Count != expectedCount)
        {
            return EmbeddingBatch.Failure(
                "the response held " +
                (root["data"] is JArray array
                    ? array.Count.ToString(CultureInfo.InvariantCulture)
                    : "no") +
                " vectors for " + expectedCount.ToString(CultureInfo.InvariantCulture) + " inputs");
        }

        var ordered = new EmbeddingVector?[expectedCount];
        for (int position = 0; position < data.Count; position++)
        {
            if (data[position] is not JObject item || item["embedding"] is not JArray components)
            {
                return EmbeddingBatch.Failure("the response held an entry with no embedding array");
            }

            int index = (int?)item["index"] ?? position;
            if (index < 0 || index >= expectedCount || ordered[index] is not null)
            {
                return EmbeddingBatch.Failure("the response held a duplicate or out-of-range vector index");
            }

            var values = new double[components.Count];
            for (int component = 0; component < components.Count; component++)
            {
                double? value = (double?)components[component];
                if (value is not { } number || double.IsNaN(number) || double.IsInfinity(number))
                {
                    return EmbeddingBatch.Failure("the response held a non-finite embedding component");
                }

                values[component] = number;
            }

            if (values.Length == 0) return EmbeddingBatch.Failure("the response held an empty embedding vector");
            ordered[index] = new EmbeddingVector(values);
        }

        var vectors = new List<EmbeddingVector>(expectedCount);
        foreach (EmbeddingVector? vector in ordered)
        {
            if (vector is null) return EmbeddingBatch.Failure("the response left an input without a vector");
            vectors.Add(vector);
        }

        return EmbeddingBatch.Success(vectors);
    }

    private JArray BuildInput(IReadOnlyList<string> texts)
    {
        var input = new JArray();
        foreach (string text in texts)
        {
            input.Add(text.Length <= MaxInputCharacters ? text : text.Substring(0, MaxInputCharacters));
        }

        return input;
    }

    private static string DescribeProblem(HttpRequestException exception)
    {
        if (exception is HttpResponseException responseException)
        {
            return "the provider returned status " +
                ((int)responseException.ResponseStatusCode).ToString(CultureInfo.InvariantCulture);
        }

        return "the request failed with " + exception.GetType().Name;
    }

    private static bool IsRetryable(HttpRequestException exception)
    {
        // Connectors throw HttpResponseException on a non-success response, preserving the status code on every
        // target framework, so classify from it first and get identical behavior everywhere.
        if (exception is HttpResponseException responseException)
        {
            int code = (int)responseException.ResponseStatusCode;
            return code == 429 || code == 408 || code >= 500;
        }

#if NET5_0_OR_GREATER
        if (exception.StatusCode is null)
        {
            // No HTTP response at all (transport or network failure) — transient, retry.
            return true;
        }

        int statusCode = (int)exception.StatusCode;
        return statusCode == 429 || statusCode == 408 || statusCode >= 500;
#else
        // .NET Framework: HttpRequestException exposes no StatusCode. Inspect the inner WebException for a
        // network-level failure (timeout/connect/DNS — transient) or an HTTP response, retrying only on 408, 429,
        // or 5xx and treating other HTTP statuses as permanent.
        for (Exception? inner = exception.InnerException; inner is not null; inner = inner.InnerException)
        {
            if (inner is System.Net.WebException webException)
            {
                if (webException.Response is System.Net.HttpWebResponse webResponse)
                {
                    int statusCode = (int)webResponse.StatusCode;
                    return statusCode == 429 || statusCode == 408 || statusCode >= 500;
                }

                switch (webException.Status)
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

        // An HttpRequestException with no classifiable inner error is a transport failure, which is transient.
        return true;
#endif
    }
}
