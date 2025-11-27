using System.Collections.Concurrent;
using AiDotNet.Inference;

namespace AiDotNet.Serving.ContinuousBatching;

/// <summary>
/// Manages continuous batching for efficient LLM inference serving.
/// </summary>
/// <remarks>
/// <para>
/// Continuous batching allows dynamic addition and removal of sequences from batches
/// at each iteration, maximizing GPU utilization and throughput. Unlike static batching
/// which waits for all sequences to complete, continuous batching can start new requests
/// as soon as others finish.
/// </para>
/// <para><b>For Beginners:</b> Continuous batching is like a well-run restaurant kitchen.
///
/// Traditional batching: Wait for full table, take all orders, prepare all at once, serve all together.
/// Continuous batching: Take orders continuously, cook as capacity allows, serve when ready.
///
/// Benefits:
/// - 2-3x higher throughput (always using full capacity)
/// - Lower latency for short requests (don't wait for long ones)
/// - Better resource utilization
///
/// The batcher coordinates:
/// 1. Receiving new generation requests
/// 2. Scheduling which sequences to process each iteration
/// 3. Running forward passes with the current batch
/// 4. Managing KV-cache for each sequence
/// 5. Detecting completed sequences and returning results
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for tensor computations.</typeparam>
public class ContinuousBatcher<T> : IDisposable where T : struct, IComparable<T>
{
    private readonly ContinuousBatcherConfig _config;
    private readonly BatchScheduler<T> _scheduler;
    private readonly KVCache<T>? _kvCache;
    private readonly Func<Tensor<T>, Tensor<T>>? _model;

    private readonly ConcurrentDictionary<long, TaskCompletionSource<GenerationResult<T>>> _pendingResults;
    private readonly ConcurrentQueue<SequenceState<T>> _incomingRequests;

    private CancellationTokenSource? _cts;
    private Task? _runLoopTask;
    private bool _isRunning;
    private bool _disposed;

    // Statistics
    private long _totalTokensGenerated;
    private long _totalRequestsProcessed;
    private long _totalIterations;
    private DateTime _startTime;

    /// <summary>
    /// Gets whether the batcher is currently running.
    /// </summary>
    public bool IsRunning => _isRunning;

    /// <summary>
    /// Gets the number of pending requests.
    /// </summary>
    public int PendingRequestCount => _pendingResults.Count;

    /// <summary>
    /// Gets the batcher configuration.
    /// </summary>
    public ContinuousBatcherConfig Config => _config;

    /// <summary>
    /// Event raised when a sequence completes generation.
    /// </summary>
    public event EventHandler<SequenceCompletedEventArgs<T>>? SequenceCompleted;

    /// <summary>
    /// Event raised when a token is generated (for streaming).
    /// </summary>
    public event EventHandler<TokenGeneratedEventArgs<T>>? TokenGenerated;

    /// <summary>
    /// Creates a new continuous batcher with the specified configuration.
    /// </summary>
    /// <param name="config">Batcher configuration.</param>
    /// <param name="model">Model forward function (input tokens -> logits).</param>
    /// <param name="kvCache">Optional KV-cache for efficient inference.</param>
    public ContinuousBatcher(
        ContinuousBatcherConfig config,
        Func<Tensor<T>, Tensor<T>>? model = null,
        KVCache<T>? kvCache = null)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _model = model;
        _kvCache = kvCache;

        _scheduler = new BatchScheduler<T>(config.SchedulerConfig);
        _pendingResults = new ConcurrentDictionary<long, TaskCompletionSource<GenerationResult<T>>>();
        _incomingRequests = new ConcurrentQueue<SequenceState<T>>();
    }

    /// <summary>
    /// Creates a new continuous batcher with default configuration.
    /// </summary>
    public ContinuousBatcher()
        : this(new ContinuousBatcherConfig())
    {
    }

    /// <summary>
    /// Submits a generation request and returns a task that completes when generation is done.
    /// </summary>
    /// <param name="request">The generation request.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Task that completes with the generation result.</returns>
    public Task<GenerationResult<T>> GenerateAsync(
        GenerationRequest<T> request,
        CancellationToken cancellationToken = default)
    {
        if (request == null)
            throw new ArgumentNullException(nameof(request));

        var sequence = new SequenceState<T>(request);
        var tcs = new TaskCompletionSource<GenerationResult<T>>(TaskCreationOptions.RunContinuationsAsynchronously);

        // Register cancellation
        cancellationToken.Register(() =>
        {
            if (_pendingResults.TryRemove(sequence.SequenceId, out _))
            {
                _scheduler.CancelSequence(sequence.SequenceId);
                tcs.TrySetCanceled(cancellationToken);
            }
        });

        _pendingResults[sequence.SequenceId] = tcs;
        _incomingRequests.Enqueue(sequence);

        // If running, the run loop will pick this up
        // If not running, start in synchronous mode
        if (!_isRunning && _config.AutoStart)
        {
            Start();
        }

        return tcs.Task;
    }

    /// <summary>
    /// Starts the continuous batching loop.
    /// </summary>
    public void Start()
    {
        if (_isRunning) return;

        _isRunning = true;
        _startTime = DateTime.UtcNow;
        _cts = new CancellationTokenSource();
        _runLoopTask = Task.Run(() => RunLoop(_cts.Token));
    }

    /// <summary>
    /// Stops the continuous batching loop.
    /// </summary>
    public async Task StopAsync()
    {
        if (!_isRunning) return;

        _isRunning = false;
        _cts?.Cancel();

        if (_runLoopTask != null)
        {
            try
            {
                await _runLoopTask.ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                // Expected
            }
        }

        _cts?.Dispose();
        _cts = null;
    }

    /// <summary>
    /// Runs a single iteration of the batching loop (for manual control).
    /// </summary>
    /// <returns>Number of tokens generated in this iteration.</returns>
    public int Step()
    {
        // Process incoming requests
        while (_incomingRequests.TryDequeue(out var sequence))
        {
            _scheduler.AddSequence(sequence);
        }

        // Get next batch
        var batch = _scheduler.ScheduleNextBatch();
        if (batch.Count == 0)
            return 0;

        _totalIterations++;
        int tokensGenerated = 0;

        // Separate prefill and decode sequences
        var prefillSequences = batch.Where(s => !s.PrefillComplete).ToList();
        var decodeSequences = batch.Where(s => s.PrefillComplete).ToList();

        // Run prefill for new sequences
        foreach (var seq in prefillSequences)
        {
            RunPrefill(seq);
        }

        // Run decode step for all sequences
        foreach (var seq in batch)
        {
            if (seq.Status == SequenceStatus.Generating)
            {
                int newToken = RunDecodeStep(seq);
                if (newToken >= 0)
                {
                    tokensGenerated++;
                    _totalTokensGenerated++;

                    // Fire token generated event
                    TokenGenerated?.Invoke(this, new TokenGeneratedEventArgs<T>
                    {
                        Sequence = seq,
                        TokenId = newToken
                    });

                    // Invoke callback if provided
                    seq.Request.OnTokenGenerated?.Invoke(newToken);

                    // Check for completion
                    if (seq.ShouldStop(_config.EosTokenId, seq.Request.StopTokenIds))
                    {
                        CompleteSequence(seq);
                    }
                }
            }
        }

        return tokensGenerated;
    }

    /// <summary>
    /// Gets current batcher statistics.
    /// </summary>
    public BatcherStatistics GetStatistics()
    {
        var schedulerStats = _scheduler.GetStatistics();
        var runtime = DateTime.UtcNow - _startTime;

        return new BatcherStatistics
        {
            TotalTokensGenerated = _totalTokensGenerated,
            TotalRequestsProcessed = _totalRequestsProcessed,
            TotalIterations = _totalIterations,
            TokensPerSecond = runtime.TotalSeconds > 0
                ? _totalTokensGenerated / runtime.TotalSeconds
                : 0,
            RequestsPerSecond = runtime.TotalSeconds > 0
                ? _totalRequestsProcessed / runtime.TotalSeconds
                : 0,
            AverageBatchSize = _totalIterations > 0
                ? (double)_totalTokensGenerated / _totalIterations
                : 0,
            WaitingRequests = schedulerStats.WaitingSequences,
            RunningRequests = schedulerStats.RunningSequences,
            MemoryUtilization = schedulerStats.MemoryUtilization,
            RuntimeSeconds = runtime.TotalSeconds
        };
    }

    private async Task RunLoop(CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                int tokensGenerated = Step();

                // If no work was done, wait a bit before checking again
                if (tokensGenerated == 0 && _scheduler.WaitingCount == 0 && _scheduler.RunningCount == 0)
                {
                    await Task.Delay(_config.IdleSleepMs, cancellationToken).ConfigureAwait(false);
                }
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                // Log error and continue
                System.Diagnostics.Debug.WriteLine($"ContinuousBatcher error: {ex.Message}");
            }
        }
    }

    private void RunPrefill(SequenceState<T> sequence)
    {
        sequence.Status = SequenceStatus.Prefilling;
        sequence.GenerationStartedAt = DateTime.UtcNow;

        if (_model != null && sequence.TokenIds.Count > 0)
        {
            // Create input tensor from prompt tokens
            var inputTokens = CreateInputTensor(sequence.TokenIds.ToArray());

            // Run model forward pass for all prompt tokens
            var logits = _model(inputTokens);

            // Get next token from logits
            int nextToken = SampleFromLogits(logits, sequence.Request);
            sequence.AppendToken(nextToken);
        }

        sequence.PrefillComplete = true;
        sequence.Status = SequenceStatus.Generating;
    }

    private int RunDecodeStep(SequenceState<T> sequence)
    {
        if (_model == null) return -1;

        // Create input tensor from last token only (incremental decoding)
        int lastToken = sequence.TokenIds[^1];
        var inputTokens = CreateInputTensor(new[] { lastToken });

        // Run model forward pass
        var logits = _model(inputTokens);

        // Sample next token
        int nextToken = SampleFromLogits(logits, sequence.Request);
        sequence.AppendToken(nextToken);

        return nextToken;
    }

    private Tensor<T> CreateInputTensor(int[] tokenIds)
    {
        // Create a simple 2D tensor [batch=1, seq_len]
        var tensor = new Tensor<T>(new[] { 1, tokenIds.Length });
        for (int i = 0; i < tokenIds.Length; i++)
        {
            tensor[new[] { 0, i }] = ConvertToT(tokenIds[i]);
        }
        return tensor;
    }

    private T ConvertToT(int value)
    {
        if (typeof(T) == typeof(float))
            return (T)(object)(float)value;
        if (typeof(T) == typeof(double))
            return (T)(object)(double)value;
        if (typeof(T) == typeof(int))
            return (T)(object)value;

        throw new NotSupportedException($"Type {typeof(T)} is not supported");
    }

    private int SampleFromLogits(Tensor<T> logits, GenerationRequest<T> request)
    {
        // Get last position logits (shape is typically [batch, seq, vocab])
        int vocabSize = logits.Shape[^1];
        int lastPos = logits.Shape.Length > 2 ? logits.Shape[^2] - 1 : 0;

        // Extract logits for sampling
        var lastLogits = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            int[] indices = logits.Shape.Length > 2
                ? new[] { 0, lastPos, i }
                : new[] { 0, i };
            lastLogits[i] = Convert.ToSingle(logits[indices]);
        }

        // Apply temperature
        if (request.Temperature != 1.0f)
        {
            for (int i = 0; i < vocabSize; i++)
            {
                lastLogits[i] /= request.Temperature;
            }
        }

        // Convert to probabilities (softmax)
        float maxLogit = lastLogits.Max();
        float sumExp = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            lastLogits[i] = (float)Math.Exp(lastLogits[i] - maxLogit);
            sumExp += lastLogits[i];
        }
        for (int i = 0; i < vocabSize; i++)
        {
            lastLogits[i] /= sumExp;
        }

        // Apply top-p (nucleus) sampling
        if (request.TopP < 1.0f)
        {
            ApplyTopP(lastLogits, request.TopP);
        }

        // Apply top-k sampling
        if (request.TopK > 0)
        {
            ApplyTopK(lastLogits, request.TopK);
        }

        // Sample from distribution
        var random = new Random();
        float r = (float)random.NextDouble();
        float cumSum = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            cumSum += lastLogits[i];
            if (cumSum >= r)
                return i;
        }

        return vocabSize - 1; // Fallback to last token
    }

    private void ApplyTopP(float[] probs, float topP)
    {
        // Sort indices by probability descending
        var indices = Enumerable.Range(0, probs.Length)
            .OrderByDescending(i => probs[i])
            .ToArray();

        float cumSum = 0;
        int cutoff = probs.Length;
        for (int i = 0; i < indices.Length; i++)
        {
            cumSum += probs[indices[i]];
            if (cumSum > topP)
            {
                cutoff = i + 1;
                break;
            }
        }

        // Zero out probabilities below cutoff
        for (int i = cutoff; i < indices.Length; i++)
        {
            probs[indices[i]] = 0;
        }

        // Renormalize
        float sum = probs.Sum();
        if (sum > 0)
        {
            for (int i = 0; i < probs.Length; i++)
            {
                probs[i] /= sum;
            }
        }
    }

    private void ApplyTopK(float[] probs, int topK)
    {
        // Sort indices by probability descending
        var indices = Enumerable.Range(0, probs.Length)
            .OrderByDescending(i => probs[i])
            .ToArray();

        // Zero out probabilities outside top-k
        for (int i = topK; i < indices.Length; i++)
        {
            probs[indices[i]] = 0;
        }

        // Renormalize
        float sum = probs.Sum();
        if (sum > 0)
        {
            for (int i = 0; i < probs.Length; i++)
            {
                probs[i] /= sum;
            }
        }
    }

    private void CompleteSequence(SequenceState<T> sequence)
    {
        sequence.Complete(sequence.FinishReason ?? StopReason.MaxLength);
        _scheduler.CompleteSequence(sequence);
        _totalRequestsProcessed++;

        // Create result
        var result = new GenerationResult<T>
        {
            SequenceId = sequence.SequenceId,
            TokenIds = sequence.TokenIds.ToList(),
            GeneratedTokens = sequence.TokenIds.Skip(sequence.PromptLength).ToList(),
            FinishReason = sequence.FinishReason ?? StopReason.MaxLength,
            GeneratedLength = sequence.GeneratedLength,
            QueueTime = sequence.QueueTime,
            GenerationTime = sequence.GenerationTime,
            TokensPerSecond = sequence.TokensPerSecond
        };

        // Complete the pending task
        if (_pendingResults.TryRemove(sequence.SequenceId, out var tcs))
        {
            tcs.TrySetResult(result);
        }

        // Fire event
        SequenceCompleted?.Invoke(this, new SequenceCompletedEventArgs<T>
        {
            Sequence = sequence,
            Result = result
        });
    }

    /// <summary>
    /// Disposes resources used by the batcher.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        StopAsync().GetAwaiter().GetResult();

        foreach (var tcs in _pendingResults.Values)
        {
            tcs.TrySetCanceled();
        }
        _pendingResults.Clear();

        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Configuration for the continuous batcher.
/// </summary>
public class ContinuousBatcherConfig
{
    /// <summary>
    /// Scheduler configuration.
    /// </summary>
    public BatchSchedulerConfig SchedulerConfig { get; set; } = new();

    /// <summary>
    /// End-of-sequence token ID.
    /// </summary>
    public int EosTokenId { get; set; } = 2;

    /// <summary>
    /// Milliseconds to sleep when idle.
    /// </summary>
    public int IdleSleepMs { get; set; } = 10;

    /// <summary>
    /// Whether to automatically start the batcher when a request is submitted.
    /// </summary>
    public bool AutoStart { get; set; } = true;

    /// <summary>
    /// Maximum number of tokens in context (prompt + generated).
    /// </summary>
    public int MaxContextLength { get; set; } = 4096;

    /// <summary>
    /// Whether to enable speculative decoding.
    /// </summary>
    public bool EnableSpeculativeDecoding { get; set; } = false;

    /// <summary>
    /// Creates config for a specific model.
    /// </summary>
    public static ContinuousBatcherConfig ForModel(string modelName, int maxBatchSize = 8)
    {
        return new ContinuousBatcherConfig
        {
            SchedulerConfig = BatchSchedulerConfig.ForModel(modelName, maxBatchSize),
            MaxContextLength = modelName.ToLowerInvariant() switch
            {
                "llama-7b" or "llama-13b" => 4096,
                "llama-70b" => 4096,
                "gpt2" => 1024,
                _ => 2048
            }
        };
    }
}

/// <summary>
/// Result of a generation request.
/// </summary>
/// <typeparam name="T">The numeric type for tensor computations.</typeparam>
public class GenerationResult<T> where T : struct, IComparable<T>
{
    /// <summary>Unique ID of the sequence.</summary>
    public long SequenceId { get; set; }

    /// <summary>All token IDs including prompt.</summary>
    public List<int> TokenIds { get; set; } = new();

    /// <summary>Only the generated tokens (excluding prompt).</summary>
    public List<int> GeneratedTokens { get; set; } = new();

    /// <summary>Reason why generation stopped.</summary>
    public StopReason FinishReason { get; set; }

    /// <summary>Number of tokens generated.</summary>
    public int GeneratedLength { get; set; }

    /// <summary>Time spent waiting in queue.</summary>
    public TimeSpan QueueTime { get; set; }

    /// <summary>Time spent generating.</summary>
    public TimeSpan? GenerationTime { get; set; }

    /// <summary>Generation speed.</summary>
    public double? TokensPerSecond { get; set; }
}

/// <summary>
/// Statistics about the batcher's operation.
/// </summary>
public class BatcherStatistics
{
    /// <summary>Total tokens generated since start.</summary>
    public long TotalTokensGenerated { get; set; }

    /// <summary>Total requests completed since start.</summary>
    public long TotalRequestsProcessed { get; set; }

    /// <summary>Total batching iterations.</summary>
    public long TotalIterations { get; set; }

    /// <summary>Tokens generated per second.</summary>
    public double TokensPerSecond { get; set; }

    /// <summary>Requests completed per second.</summary>
    public double RequestsPerSecond { get; set; }

    /// <summary>Average batch size per iteration.</summary>
    public double AverageBatchSize { get; set; }

    /// <summary>Requests currently waiting.</summary>
    public int WaitingRequests { get; set; }

    /// <summary>Requests currently being processed.</summary>
    public int RunningRequests { get; set; }

    /// <summary>Memory utilization (0-1).</summary>
    public double MemoryUtilization { get; set; }

    /// <summary>Total runtime in seconds.</summary>
    public double RuntimeSeconds { get; set; }
}

/// <summary>
/// Event args for sequence completion.
/// </summary>
public class SequenceCompletedEventArgs<T> : EventArgs where T : struct, IComparable<T>
{
    /// <summary>The completed sequence.</summary>
    public required SequenceState<T> Sequence { get; set; }

    /// <summary>The generation result.</summary>
    public required GenerationResult<T> Result { get; set; }
}

/// <summary>
/// Event args for token generation.
/// </summary>
public class TokenGeneratedEventArgs<T> : EventArgs where T : struct, IComparable<T>
{
    /// <summary>The sequence that generated the token.</summary>
    public required SequenceState<T> Sequence { get; set; }

    /// <summary>The generated token ID.</summary>
    public required int TokenId { get; set; }
}
