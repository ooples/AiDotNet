using System.Net;
using System.Net.Http;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>
/// Options every hosted text-to-speech client shares: credentials, the endpoint, the voice, the requested sample rate,
/// retries and the time limit of a request.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A hosted voice service synthesizes speech on its own servers; these options say which
/// account (API key), which server (endpoint), which voice and what audio format to ask for.</para>
/// </remarks>
public abstract class CloudTtsOptions
{
    /// <summary>Initializes the shared defaults.</summary>
    protected CloudTtsOptions()
    {
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    protected CloudTtsOptions(CloudTtsOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        ApiKey = other.ApiKey;
        ApiEndpoint = other.ApiEndpoint;
        VoiceId = other.VoiceId;
        SampleRate = other.SampleRate;
        MaxTextLength = other.MaxTextLength;
        MaxRetries = other.MaxRetries;
        InitialRetryDelayMs = other.InitialRetryDelayMs;
        TimeoutSeconds = other.TimeoutSeconds;
    }

    /// <summary>Gets or sets the service's API key.</summary>
    public string ApiKey { get; set; } = string.Empty;

    /// <summary>Gets or sets the service's base URL; empty uses the service's public endpoint.</summary>
    public string ApiEndpoint { get; set; } = string.Empty;

    /// <summary>Gets or sets the voice to synthesize with (the service's identifier).</summary>
    public string VoiceId { get; set; } = string.Empty;

    /// <summary>Gets or sets the sample rate to request, in Hz.</summary>
    public int SampleRate { get; set; } = 24000;

    /// <summary>Gets or sets the longest text the service accepts in one request.</summary>
    public int MaxTextLength { get; set; } = 5000;

    /// <summary>Gets or sets the retries after a rate limit (429) or a server error (5xx).</summary>
    public int MaxRetries { get; set; } = 3;

    /// <summary>Gets or sets the first retry delay in milliseconds (doubled per retry unless the service sends
    /// Retry-After).</summary>
    public int InitialRetryDelayMs { get; set; } = 1000;

    /// <summary>Gets or sets the time limit of one request, in seconds.</summary>
    public double TimeoutSeconds { get; set; } = 120;
}

/// <summary>
/// The failure of a hosted text-to-speech request: the service, its HTTP status and the start of its error body.
/// </summary>
public sealed class CloudTtsException : Exception
{
    /// <summary>Creates the exception.</summary>
    public CloudTtsException(string provider, HttpStatusCode statusCode, string responseBody)
        : base($"{provider} returned HTTP {(int)statusCode} ({statusCode}): {Truncate(responseBody)}")
    {
        Provider = provider;
        StatusCode = statusCode;
        ResponseBody = responseBody;
    }

    /// <summary>The service that failed.</summary>
    public string Provider { get; }

    /// <summary>The HTTP status code.</summary>
    public HttpStatusCode StatusCode { get; }

    /// <summary>The response body.</summary>
    public string ResponseBody { get; }

    private static string Truncate(string body) => body.Length <= 500 ? body : body.Substring(0, 500) + "…";
}

/// <summary>
/// A client of a hosted text-to-speech service: it sends the text to the service's REST API and decodes the audio it
/// returns into a waveform tensor.
/// </summary>
/// <typeparam name="T">The numeric type of the returned waveform.</typeparam>
/// <remarks>
/// <para>
/// These services are proprietary systems with no published architecture to reproduce, so the faithful implementation is
/// a client of the service itself. A request is retried after a rate limit (429) or a server error (5xx), honouring the
/// service's Retry-After header, otherwise after an exponentially growing delay. The waveform is mono, in [−1, 1], at
/// the sample rate the service was asked for.
/// </para>
/// <para><b>For Beginners:</b> Call <see cref="Synthesize"/> with your text; the class sends it to the vendor over the
/// internet with your API key and gives you back the audio samples.</para>
/// </remarks>
public abstract class CloudTtsClientBase<T> : ITtsModel<T>, IDisposable
{
    /// <summary>Numeric operations for <typeparamref name="T"/>.</summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly bool _ownsClient;
    private bool _disposed;

    /// <summary>Creates the client; a supplied <see cref="System.Net.Http.HttpClient"/> is used as is and not disposed.</summary>
    protected CloudTtsClientBase(CloudTtsOptions options, HttpClient? httpClient)
    {
        Settings = options ?? throw new ArgumentNullException(nameof(options));
        _ownsClient = httpClient is null;
        HttpClient = httpClient ?? new HttpClient { Timeout = System.Threading.Timeout.InfiniteTimeSpan };
    }

    /// <summary>The HTTP client requests go through.</summary>
    protected HttpClient HttpClient { get; }

    /// <summary>The client's options.</summary>
    protected CloudTtsOptions Settings { get; }

    /// <summary>The service's name, used in messages.</summary>
    public abstract string ProviderName { get; }

    /// <inheritdoc />
    public int SampleRate => Settings.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => Settings.MaxTextLength;

    /// <summary>Throws when the options cannot make a request (missing credentials, an unsupported sample rate).</summary>
    protected abstract void ValidateConfiguration();

    /// <summary>The request that synthesizes <paramref name="text"/>.</summary>
    protected abstract HttpRequestMessage CreateRequest(string text);

    /// <summary>The waveform <c>[samples]</c> of a successful response's body.</summary>
    protected abstract Tensor<T> DecodeAudio(byte[] body, string? mediaType);

    /// <inheritdoc />
    /// <remarks>Blocks on <see cref="SynthesizeAsync"/>.</remarks>
    public Tensor<T> Synthesize(string text) => SynthesizeAsync(text).ConfigureAwait(false).GetAwaiter().GetResult();

    /// <summary>Synthesizes <paramref name="text"/> through the service.</summary>
    /// <param name="text">The text to speak.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <returns>The mono waveform <c>[samples]</c> in [−1, 1] at <see cref="SampleRate"/>.</returns>
    public async Task<Tensor<T>> SynthesizeAsync(string text, CancellationToken cancellationToken = default)
    {
        if (_disposed) throw new ObjectDisposedException(GetType().Name);
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Text is required.", nameof(text));
        if (text.Length > Settings.MaxTextLength)
            throw new ArgumentException($"{ProviderName} accepts at most {Settings.MaxTextLength} characters per request; got {text.Length}.", nameof(text));
        ValidateConfiguration();

        int delay = Math.Max(0, Settings.InitialRetryDelayMs);
        for (int attempt = 0; ; attempt++)
        {
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeout.CancelAfter(TimeSpan.FromSeconds(Settings.TimeoutSeconds));
            using var request = CreateRequest(text);
            HttpResponseMessage response;
            try
            {
                response = await HttpClient.SendAsync(request, timeout.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested && attempt < Settings.MaxRetries)
            {
                await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
                delay *= 2;
                continue;
            }
            using (response)
            {
                int status = (int)response.StatusCode;
                if ((status == 429 || status >= 500) && attempt < Settings.MaxRetries)
                {
                    var retryAfter = response.Headers.RetryAfter;
                    double requested = retryAfter?.Delta is TimeSpan d ? d.TotalMilliseconds
                        : retryAfter?.Date is DateTimeOffset at ? (at - DateTimeOffset.UtcNow).TotalMilliseconds
                        : delay;
                    // A server's Retry-After is honoured up to the request timeout: a longer one (or one past int's range)
                    // would otherwise stall the call far beyond what the caller configured.
                    int wait = (int)Math.Min(Math.Max(0, requested), Math.Min(Settings.TimeoutSeconds * 1000.0, int.MaxValue));
                    await Task.Delay(wait, cancellationToken).ConfigureAwait(false);
                    delay *= 2;
                    continue;
                }
                var body = await response.Content.ReadAsByteArrayAsync().ConfigureAwait(false);
                if (!response.IsSuccessStatusCode)
                    throw new CloudTtsException(ProviderName, response.StatusCode, System.Text.Encoding.UTF8.GetString(body));
                return DecodeAudio(body, response.Content.Headers.ContentType?.MediaType);
            }
        }
    }

    /// <summary>The service's base URL: the configured endpoint, or <paramref name="defaultEndpoint"/>.</summary>
    protected string BaseUrl(string defaultEndpoint)
        => (string.IsNullOrWhiteSpace(Settings.ApiEndpoint) ? defaultEndpoint : Settings.ApiEndpoint).TrimEnd('/');

    /// <summary>Throws when the API key is missing.</summary>
    protected void RequireApiKey()
    {
        if (string.IsNullOrWhiteSpace(Settings.ApiKey))
            throw new InvalidOperationException($"{ProviderName} needs an API key: set {nameof(CloudTtsOptions.ApiKey)}.");
    }

    /// <summary>Throws when <see cref="CloudTtsOptions.SampleRate"/> is not one of <paramref name="supported"/>.</summary>
    protected void RequireSampleRate(params int[] supported)
    {
        if (Array.IndexOf(supported, Settings.SampleRate) < 0)
            throw new InvalidOperationException(
                $"{ProviderName} returns uncompressed audio at {string.Join(", ", supported)} Hz; {Settings.SampleRate} Hz is not one of them.");
    }

    /// <summary>The waveform of raw signed 16-bit little-endian mono PCM.</summary>
    protected static Tensor<T> FromPcm16(byte[] pcm)
    {
        int n = pcm.Length / 2;
        var wave = new Tensor<T>(new[] { n });
        for (int i = 0; i < n; i++) wave[i] = NumOps.FromDouble((short)(pcm[2 * i] | (pcm[2 * i + 1] << 8)) / 32768.0);
        return wave;
    }

    /// <summary>The waveform of a WAV file, mixed to mono; throws when its sample rate is not the one requested.</summary>
    protected Tensor<T> FromWav(byte[] wav)
    {
        var decoded = AudioHelper<T>.DecodeWav(wav);
        if (decoded.SampleRate != Settings.SampleRate)
            throw new InvalidDataException($"{ProviderName} returned {decoded.SampleRate} Hz audio; {Settings.SampleRate} Hz was requested.");
        return Mono(decoded.Audio);
    }

    /// <summary>The mean over channels of <c>[1, channels, samples]</c>, as <c>[samples]</c>.</summary>
    protected static Tensor<T> Mono(Tensor<T> audio)
    {
        int channels = audio.Shape[1], n = audio.Shape[2];
        var wave = new Tensor<T>(new[] { n });
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int c = 0; c < channels; c++) sum += NumOps.ToDouble(audio[0, c, i]);
            wave[i] = NumOps.FromDouble(sum / channels);
        }
        return wave;
    }

    /// <summary>Releases the HTTP client when the client created it.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_ownsClient) HttpClient.Dispose();
        GC.SuppressFinalize(this);
    }
}
