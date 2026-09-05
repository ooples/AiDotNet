using System.Net.Http;
using System.Threading;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Models.Connectors;
using AiDotNet.Evolution;
using AiDotNet.Validation;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>Retries transient chat failures with bounded exponential backoff, jitter, and a per-call timeout.</summary>
/// <remarks>
/// <para>
/// Retry lives here, in the pipeline, rather than inside one connector base class. That single change makes the
/// policy apply to every backend the library can talk to — a local engine, an Ollama endpoint, a Microsoft
/// Extensions AI bridge, a recorded replay client — instead of only to the HTTP connectors that happen to derive
/// from a particular base. It also makes the policy configurable and observable, which a protected field on a
/// base class is not.
/// </para>
/// <para>
/// A call is attempted at most <c>MaxRetries + 1</c> times, and that is a guarantee rather than an
/// approximation. The reference OpenEvolve implementation configures retries in its own loop <em>and</em> passes
/// the same number to the provider SDK's internal retry, so a configured value of three can produce up to sixteen
/// HTTP attempts against a rate-limited endpoint — the exact circumstance where retrying hardest is most harmful.
/// Delay grows exponentially from the base delay, is capped, and carries jitter drawn from a seeded
/// <see cref="StableRandom"/>, so a fleet of workers that all fail at once does not resynchronize on the retry.
/// </para>
/// <para>
/// Only transient conditions are retried: transport failures, request timeouts, and HTTP 408, 429, and 5xx. A
/// rejected key or a malformed request is returned immediately, because retrying it wastes the budget and delays
/// the error the caller needs to see. Caller cancellation is never treated as a failure.
/// </para>
/// <para><b>For Beginners:</b> Network calls to AI providers fail sometimes — a connection drops, the service is
/// briefly overloaded, a request takes too long. This wrapper quietly tries again a few times, waiting a little
/// longer between attempts, and gives up with a clear error if the problem persists. It deliberately does not
/// retry mistakes that will never succeed, such as a wrong API key. Add it once and every model call in your
/// application gets the same behaviour.</para>
/// </remarks>
public sealed class RetryChatMiddleware : IChatMiddleware
{
    /// <summary>The default number of retries after the first attempt.</summary>
    public const int DefaultMaxRetries = 3;

    /// <summary>The largest number of retries accepted.</summary>
    public const int MaxSupportedRetries = 16;

    private readonly object _gate = new();
    private readonly StableRandom _jitterSource;
    private readonly Func<Exception, bool>? _shouldRetry;
    private long _totalAttempts;
    private long _totalRetries;
    private long _totalTimeouts;

    /// <summary>Initializes a retry policy.</summary>
    /// <param name="maxRetries">Retries after the first attempt; <c>0</c> disables retrying.</param>
    /// <param name="baseDelay">The wait before the first retry, doubling thereafter. <c>null</c> uses one second.</param>
    /// <param name="maxDelay">The ceiling on any single wait. <c>null</c> uses thirty seconds.</param>
    /// <param name="callTimeout">The time limit for one attempt. <c>null</c> means no per-attempt limit.</param>
    /// <param name="jitterFactor">The fraction of each wait that is randomized, from 0 to 1.</param>
    /// <param name="seed">The seed for the jitter stream, so a run's waits are reproducible.</param>
    /// <param name="shouldRetry">
    /// An extra predicate consulted for exceptions the built-in classification does not consider transient, or
    /// <c>null</c> to use the built-in classification alone.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="maxRetries"/> is negative or above <see cref="MaxSupportedRetries"/>, a delay is negative
    /// or not finite, <paramref name="maxDelay"/> is smaller than <paramref name="baseDelay"/>, or
    /// <paramref name="jitterFactor"/> is outside 0 to 1.
    /// </exception>
    public RetryChatMiddleware(
        int maxRetries = DefaultMaxRetries,
        TimeSpan? baseDelay = null,
        TimeSpan? maxDelay = null,
        TimeSpan? callTimeout = null,
        double jitterFactor = 0.25,
        ulong seed = 0x5EED_1234_5678_9ABCUL,
        Func<Exception, bool>? shouldRetry = null)
    {
        if (maxRetries < 0 || maxRetries > MaxSupportedRetries)
        {
            throw new ArgumentOutOfRangeException(nameof(maxRetries), maxRetries,
                $"Value must be between 0 and {MaxSupportedRetries}.");
        }

        TimeSpan resolvedBase = baseDelay ?? TimeSpan.FromSeconds(1);
        TimeSpan resolvedMax = maxDelay ?? TimeSpan.FromSeconds(30);
        RequireNonNegative(resolvedBase, nameof(baseDelay));
        RequireNonNegative(resolvedMax, nameof(maxDelay));
        if (resolvedMax < resolvedBase)
        {
            throw new ArgumentOutOfRangeException(nameof(maxDelay), resolvedMax,
                "The maximum delay cannot be smaller than the base delay.");
        }

        if (callTimeout.HasValue)
        {
            if (callTimeout.Value <= TimeSpan.Zero)
            {
                throw new ArgumentOutOfRangeException(nameof(callTimeout), callTimeout.Value,
                    "A call timeout must be positive.");
            }
        }

        if (double.IsNaN(jitterFactor) || double.IsInfinity(jitterFactor) || jitterFactor < 0 || jitterFactor > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(jitterFactor), jitterFactor,
                "Value must be a finite number between 0 and 1.");
        }

        MaxRetries = maxRetries;
        BaseDelay = resolvedBase;
        MaxDelay = resolvedMax;
        CallTimeout = callTimeout;
        JitterFactor = jitterFactor;
        _jitterSource = new StableRandom(seed);
        _shouldRetry = shouldRetry;
    }

    /// <summary>Gets the number of retries attempted after the first attempt.</summary>
    public int MaxRetries { get; }

    /// <summary>Gets the wait before the first retry.</summary>
    public TimeSpan BaseDelay { get; }

    /// <summary>Gets the ceiling on any single wait.</summary>
    public TimeSpan MaxDelay { get; }

    /// <summary>Gets the time limit for one attempt, or <c>null</c> when attempts are not time-limited.</summary>
    public TimeSpan? CallTimeout { get; }

    /// <summary>Gets the fraction of each wait that is randomized.</summary>
    public double JitterFactor { get; }

    /// <summary>Gets how many attempts, including first attempts, this policy has made.</summary>
    public long TotalAttempts => Interlocked.Read(ref _totalAttempts);

    /// <summary>Gets how many of those attempts were retries.</summary>
    public long TotalRetries => Interlocked.Read(ref _totalRetries);

    /// <summary>Gets how many attempts exceeded <see cref="CallTimeout"/>.</summary>
    public long TotalTimeouts => Interlocked.Read(ref _totalTimeouts);

    /// <inheritdoc/>
    public async Task<ChatResponse> InvokeAsync(
        ChatRequestContext context,
        ChatPipelineDelegate next,
        CancellationToken cancellationToken)
    {
        Guard.NotNull(context);
        Guard.NotNull(next);

        Exception? lastError = null;
        for (int attempt = 0; attempt <= MaxRetries; attempt++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Interlocked.Increment(ref _totalAttempts);
            if (attempt > 0) Interlocked.Increment(ref _totalRetries);

            try
            {
                return await InvokeOnceAsync(context, next, cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                // The caller asked to stop. That is not a failure to retry.
                throw;
            }
            catch (Exception exception) when (IsTransient(exception))
            {
                lastError = exception;
                if (exception is TimeoutException) Interlocked.Increment(ref _totalTimeouts);
                if (attempt == MaxRetries) break;
                await Task.Delay(NextDelay(attempt), cancellationToken).ConfigureAwait(false);
            }
        }

        throw new InvalidOperationException(
            $"The chat call did not succeed after {MaxRetries + 1} attempt(s). " +
            $"The last failure was {lastError?.GetType().Name ?? "unknown"}.",
            lastError);
    }

    /// <summary>Classifies an exception as a transient condition worth retrying.</summary>
    /// <param name="exception">The exception to classify.</param>
    /// <returns><c>true</c> when another attempt could plausibly succeed.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="exception"/> is <c>null</c>.</exception>
    public bool IsTransient(Exception exception)
    {
        Guard.NotNull(exception);

        if (exception is HttpResponseException responseException)
        {
            int code = (int)responseException.ResponseStatusCode;
            return code == 408 || code == 429 || code >= 500;
        }

        if (exception is TimeoutException) return true;
        if (exception is TaskCanceledException) return true;
        if (exception is IOException) return true;
        if (exception is HttpRequestException httpException) return IsTransientHttp(httpException);

        return _shouldRetry is not null && _shouldRetry(exception);
    }

    private async Task<ChatResponse> InvokeOnceAsync(
        ChatRequestContext context,
        ChatPipelineDelegate next,
        CancellationToken cancellationToken)
    {
        if (CallTimeout is not { } timeout)
        {
            return await next(context, cancellationToken).ConfigureAwait(false);
        }

        using var timer = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timer.CancelAfter(timeout);
        try
        {
            return await next(context, timer.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested && timer.IsCancellationRequested)
        {
            // Our own deadline fired, not the caller's token. Surface it as a
            // timeout so the retry classification is unambiguous and so a caller
            // that catches it cannot mistake it for their own cancellation.
            throw new TimeoutException(
                $"The chat call exceeded the configured timeout of {timeout.TotalSeconds:0.###} seconds.");
        }
    }

    private TimeSpan NextDelay(int attempt)
    {
        double scale = Math.Pow(2, attempt);
        double milliseconds = BaseDelay.TotalMilliseconds * scale;
        double capped = Math.Min(milliseconds, MaxDelay.TotalMilliseconds);
        if (JitterFactor <= 0 || capped <= 0) return TimeSpan.FromMilliseconds(capped);

        double fraction;
        lock (_gate)
        {
            // One generator shared by concurrent calls, so the lock is required;
            // the sequence stays reproducible for a single-threaded run, which is
            // what a deterministic test needs.
            fraction = _jitterSource.NextDouble();
        }

        double jitter = capped * JitterFactor * ((fraction * 2.0) - 1.0);
        double withJitter = Math.Max(0, capped + jitter);
        return TimeSpan.FromMilliseconds(withJitter);
    }

    private static bool IsTransientHttp(HttpRequestException exception)
    {
#if NET5_0_OR_GREATER
        if (exception.StatusCode is null) return true;
        int statusCode = (int)exception.StatusCode;
        return statusCode == 408 || statusCode == 429 || statusCode >= 500;
#else
        for (Exception? inner = exception.InnerException; inner is not null; inner = inner.InnerException)
        {
            if (inner is System.Net.WebException webException)
            {
                if (webException.Response is System.Net.HttpWebResponse response)
                {
                    int statusCode = (int)response.StatusCode;
                    return statusCode == 408 || statusCode == 429 || statusCode >= 500;
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

        return true;
#endif
    }

    private static void RequireNonNegative(TimeSpan value, string parameterName)
    {
        if (value < TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(parameterName, value, "A delay cannot be negative.");
        }
    }
}
