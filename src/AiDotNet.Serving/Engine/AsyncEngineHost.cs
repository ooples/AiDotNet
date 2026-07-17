using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

namespace AiDotNet.Serving.Engine;

/// <summary>
/// One incremental output for a streamed request: the tokens generated so far, whether it has finished, and
/// why. The async host emits one of these per engine step for each in-flight request.
/// </summary>
public readonly struct GenerationUpdate
{
    /// <summary>Creates an update.</summary>
    public GenerationUpdate(IReadOnlyList<int> tokenIds, bool isFinished, string? finishReason)
    {
        TokenIds = tokenIds;
        IsFinished = isFinished;
        FinishReason = finishReason;
    }

    /// <summary>All generated token ids so far (cumulative, excludes the prompt).</summary>
    public IReadOnlyList<int> TokenIds { get; }

    /// <summary>True once the request has finished.</summary>
    public bool IsFinished { get; }

    /// <summary>Reason it stopped ("stop", "length", "abort"), or null while running.</summary>
    public string? FinishReason { get; }
}

/// <summary>
/// Runs a <see cref="ContinuousBatchingEngine{T}"/> on a dedicated pump thread and exposes an async, concurrent
/// submission API on top of it — the piece that lets many callers (or many HTTP connections) generate at the
/// same time while a single loop advances the shared batch. This is the concurrency core beneath the HTTP
/// server; it has no transport dependency and is unit-testable directly.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> the engine itself advances one step at a time and must be driven from a single
/// loop. This class runs that loop for you on a background thread and hands each caller an async result (or a
/// live stream of tokens), so your web requests don't have to know anything about stepping the engine.</para>
/// </remarks>
/// <typeparam name="T">The model's numeric type.</typeparam>
public sealed class AsyncEngineHost<T> : IDisposable
{
    private readonly IInferenceEngine _engine;
    private readonly Thread _pumpThread;
    private readonly SemaphoreSlim _wakeup = new(0);
    private readonly ConcurrentDictionary<string, Channel<GenerationUpdate>> _sinks = new();
    private volatile bool _running = true;
    private long _counter;

    /// <summary>Wraps and starts a pump over the given engine.</summary>
    public AsyncEngineHost(IInferenceEngine engine)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _pumpThread = new Thread(PumpLoop) { IsBackground = true, Name = "aidotnet-serving-pump" };
        _pumpThread.Start();
    }

    /// <summary>A snapshot of engine load and KV-cache utilization.</summary>
    public EngineStatistics GetStatistics() => _engine.GetStatistics();

    /// <summary>
    /// Streams generation updates for a prompt as tokens are produced. Cancelling the enumeration aborts the
    /// request in the engine and frees its KV memory.
    /// </summary>
    public async IAsyncEnumerable<GenerationUpdate> StreamAsync(
        IReadOnlyList<int> promptTokenIds,
        SamplingParameters sampling,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        string id = Submit(promptTokenIds, sampling, out var reader);
        try
        {
            while (await reader.WaitToReadAsync(cancellationToken).ConfigureAwait(false))
            {
                while (reader.TryRead(out var update))
                {
                    yield return update;
                    if (update.IsFinished) yield break;
                }
            }
        }
        finally
        {
            if (_sinks.TryRemove(id, out _)) _engine.AbortRequest(id);
        }
    }

    /// <summary>Generates a complete continuation for a prompt, returning the generated token ids.</summary>
    public async Task<IReadOnlyList<int>> GenerateAsync(
        IReadOnlyList<int> promptTokenIds,
        SamplingParameters sampling,
        CancellationToken cancellationToken = default)
    {
        IReadOnlyList<int> result = Array.Empty<int>();
        await foreach (var update in StreamAsync(promptTokenIds, sampling, cancellationToken).ConfigureAwait(false))
            result = update.TokenIds;
        return result;
    }

    private string Submit(IReadOnlyList<int> promptTokenIds, SamplingParameters sampling, out ChannelReader<GenerationUpdate> reader)
    {
        if (promptTokenIds is null) throw new ArgumentNullException(nameof(promptTokenIds));
        if (sampling is null) throw new ArgumentNullException(nameof(sampling));

        string id = "req-" + Interlocked.Increment(ref _counter).ToString();
        var channel = Channel.CreateUnbounded<GenerationUpdate>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
        _sinks[id] = channel;
        reader = channel.Reader;

        _engine.AddRequest(new GenerationRequest(id, promptTokenIds, sampling));
        _wakeup.Release();
        return id;
    }

    private void PumpLoop()
    {
        while (_running)
        {
            IReadOnlyList<RequestOutput> outputs;
            try
            {
                outputs = _engine.Step();
            }
            catch (Exception ex)
            {
                FailAllSinks(ex);
                continue;
            }

            foreach (var output in outputs)
                Dispatch(output);

            if (outputs.Count == 0 && !_engine.HasUnfinishedRequests)
                _wakeup.Wait(100); // sleep until a new request arrives (bounded so shutdown stays responsive)
        }
    }

    private void Dispatch(RequestOutput output)
    {
        if (!_sinks.TryGetValue(output.RequestId, out var channel)) return;

        var completion = output.Outputs.Count > 0 ? output.Outputs[0] : null;
        var tokens = completion?.TokenIds ?? Array.Empty<int>();
        channel.Writer.TryWrite(new GenerationUpdate(tokens, output.IsFinished, completion?.FinishReason));

        if (output.IsFinished)
        {
            channel.Writer.TryComplete();
            _sinks.TryRemove(output.RequestId, out _);
        }
    }

    private void FailAllSinks(Exception ex)
    {
        foreach (var kvp in _sinks)
        {
            kvp.Value.Writer.TryComplete(ex);
            _sinks.TryRemove(kvp.Key, out _);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _running = false;
        _wakeup.Release();
        if (_pumpThread.IsAlive) _pumpThread.Join(TimeSpan.FromSeconds(5));
        foreach (var kvp in _sinks) kvp.Value.Writer.TryComplete();
        _engine.Dispose();
    }
}
