using System.Collections.Concurrent;
using AiDotNet.Helpers;
using AiDotNet.Inference;
using AiDotNet.Inference.SpeculativeDecoding;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Validation;

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
internal class ContinuousBatcher<T> : IDisposable
{
    private readonly ContinuousBatcherConfig _config;
    private readonly BatchScheduler<T> _scheduler;
    private readonly KVCache<T>? _kvCache;
    private readonly Func<Tensor<T>, Tensor<T>>? _model;
    private readonly IDraftModel<T>? _draftModelOverride;

    private SpeculativeDecoder<T>? _speculativeDecoder;
    private readonly object _speculativeLock = new();
    private volatile bool _speculationDisabledDueToFailure;
    private long _speculationDisabledUntilIteration;

    internal bool LastStepUsedSpeculation { get; private set; }
    internal int LastStepSpeculationTokens { get; private set; }
    internal string LastStepSpeculationReason { get; private set; } = string.Empty;

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
        KVCache<T>? kvCache = null,
        IDraftModel<T>? draftModel = null)
    {
        Guard.NotNull(config);
        _config = config;
        _model = model;
        _kvCache = kvCache;
        _draftModelOverride = draftModel;

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

        bool useSpeculation = ShouldUseSpeculativeDecoding(batch, out var speculationReason);
        LastStepUsedSpeculation = useSpeculation;
        LastStepSpeculationTokens = 0;
        LastStepSpeculationReason = speculationReason;

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
                var newTokens = useSpeculation ? RunDecodeStepSpeculative(seq) : RunDecodeStep(seq);
                if (newTokens.Count > 0)
                {
                    foreach (var newToken in newTokens)
                    {
                        tokensGenerated++;
                        _totalTokensGenerated++;
                        if (useSpeculation)
                            LastStepSpeculationTokens++;

                        // Fire token generated event
                        TokenGenerated?.Invoke(this, new TokenGeneratedEventArgs<T>
                        {
                            Sequence = seq,
                            TokenId = newToken
                        });

                        // Invoke callback if provided
                        seq.Request.OnTokenGenerated?.Invoke(newToken);

                        // Check for completion after each appended token
                        if (seq.ShouldStop(_config.EosTokenId, seq.Request.StopTokenIds))
                        {
                            CompleteSequence(seq);
                            break;
                        }
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

    private IReadOnlyList<int> RunDecodeStep(SequenceState<T> sequence)
    {
        if (_model == null) return Array.Empty<int>();

        // Create input tensor from last token only (incremental decoding)
        int lastToken = sequence.TokenIds[^1];
        var inputTokens = CreateInputTensor([lastToken]);

        // Run model forward pass
        var logits = _model(inputTokens);

        // Sample next token
        int nextToken = SampleFromLogits(logits, sequence.Request);
        sequence.AppendToken(nextToken);

        return new[] { nextToken };
    }

    private IReadOnlyList<int> RunDecodeStepSpeculative(SequenceState<T> sequence)
    {
        if (_model == null) return Array.Empty<int>();
        if (!ShouldSpeculateForThisIteration()) return RunDecodeStep(sequence);

        int remaining = sequence.MaxNewTokens - sequence.GeneratedLength;
        if (remaining <= 0) return Array.Empty<int>();

        var decoder = EnsureSpeculativeDecoder();
        if (decoder == null) return RunDecodeStep(sequence);

        var numOps = MathHelper.GetNumericOperations<T>();
        T temperature = numOps.FromDouble(sequence.Request.Temperature);

        var inputTokens = new Vector<int>(sequence.TokenIds.ToArray());
        int maxNew = Math.Min(remaining, Math.Max(1, _config.SpeculationDepth + 1));

        SpeculativeResult result;
        try
        {
            result = decoder.Generate(
                inputTokens,
                maxNewTokens: maxNew,
                temperature: temperature,
                eosToken: _config.EosTokenId);
        }
        catch (Exception ex)
        {
            _speculationDisabledDueToFailure = true;
            InferenceDiagnostics.RecordException(
                area: "Serving.ContinuousBatching",
                feature: "SpeculativeDecoding",
                ex: ex,
                reason: "Speculative decoder execution failed; falling back to baseline decode.");
            InferenceDiagnostics.RecordDecision(
                area: "Serving.ContinuousBatching",
                feature: "SpeculativeDecoding",
                enabled: false,
                reason: "DisabledDueToFailure");
            return RunDecodeStep(sequence);
        }

        if (result.NewTokens.Length == 0)
            return Array.Empty<int>();

        var tokens = new List<int>(result.NewTokens.Length);
        for (int i = 0; i < result.NewTokens.Length; i++)
        {
            int token = result.NewTokens[i];
            sequence.AppendToken(token);
            tokens.Add(token);

            // Prevent appending beyond stop conditions (e.g., EOS in the speculative batch).
            if (sequence.ShouldStop(_config.EosTokenId, sequence.Request.StopTokenIds))
            {
                break;
            }
        }

        return tokens;
    }

    private bool ShouldUseSpeculativeDecoding(IReadOnlyCollection<SequenceState<T>> batch, out string reason)
    {
        if (_speculationDisabledDueToFailure)
        {
            reason = "DisabledDueToFailure";
            InferenceDiagnostics.RecordDecision("Serving.ContinuousBatching", "SpeculativeDecoding", enabled: false, reason: reason);
            return false;
        }

        if (!_config.EnableSpeculativeDecoding)
        {
            reason = "DisabledByConfig";
            InferenceDiagnostics.RecordDecision("Serving.ContinuousBatching", "SpeculativeDecoding", enabled: false, reason: reason);
            return false;
        }

        if (_config.SpeculationPolicy == AiDotNet.Configuration.SpeculationPolicy.ForceOff)
        {
            reason = "ForceOff";
            InferenceDiagnostics.RecordDecision("Serving.ContinuousBatching", "SpeculativeDecoding", enabled: false, reason: reason);
            return false;
        }

        if (_config.SpeculationPolicy == AiDotNet.Configuration.SpeculationPolicy.ForceOn)
        {
            reason = "ForceOn";
            InferenceDiagnostics.RecordDecision("Serving.ContinuousBatching", "SpeculativeDecoding", enabled: true, reason: reason);
            return true;
        }

        if (_config.SpeculationPolicy == AiDotNet.Configuration.SpeculationPolicy.ThroughputFirst)
        {
            // Extremely conservative: only speculate when there is no queue pressure and batches are tiny.
            bool ok = batch.Count == 1 && _scheduler.WaitingCount == 0 && _speculationDisabledUntilIteration <= _totalIterations;
            reason = ok ? "ThroughputFirst(Enabled)" : "ThroughputFirst(Backoff)";
            InferenceDiagnostics.RecordDecision("Serving.ContinuousBatching", "SpeculativeDecoding", enabled: ok, reason: reason);
            return ok;
        }

        // Auto policy: back off under load and when draft acceptance is too low.
        if (_speculationDisabledUntilIteration > _totalIterations)
        {
            reason = "AutoBackoff(Cooldown)";
            InferenceDiagnostics.RecordDecision("Serving.ContinuousBatching", "SpeculativeDecoding", enabled: false, reason: reason);
            return false;
        }

        int maxBatchForSpeculation = _config.SchedulerConfig.MaxBatchSize / 2;
        if (_config.SpeculationPolicy == AiDotNet.Configuration.SpeculationPolicy.LatencyFirst)
        {
            // Allow more speculation under load, but still avoid it when the queue is growing.
            maxBatchForSpeculation = Math.Max(1, _config.SchedulerConfig.MaxBatchSize);
        }

        bool enabled = batch.Count <= Math.Max(1, maxBatchForSpeculation) && _scheduler.WaitingCount == 0;
        if (!enabled)
        {
            reason = _config.SpeculationPolicy == AiDotNet.Configuration.SpeculationPolicy.LatencyFirst
                ? "LatencyFirst(Backoff:LoadOrQueue)"
                : "AutoBackoff(LoadOrQueue)";
            InferenceDiagnostics.RecordDecision("Serving.ContinuousBatching", "SpeculativeDecoding", enabled: false, reason: reason);
            return false;
        }

        // If we have enough evidence that the draft model is low-quality, disable speculation for a short cooldown.
        var decoder = _speculativeDecoder;
        if (decoder != null && decoder.TotalDraftTokens >= 32 && decoder.AcceptanceRate < 0.25)
        {
            _speculationDisabledUntilIteration = _totalIterations + 25;
            reason = $"AutoBackoff(LowAcceptanceRate={decoder.AcceptanceRate:0.00})";
            InferenceDiagnostics.RecordDecision("Serving.ContinuousBatching", "SpeculativeDecoding", enabled: false, reason: reason);
            return false;
        }

        reason = "AutoEnabled";
        InferenceDiagnostics.RecordDecision("Serving.ContinuousBatching", "SpeculativeDecoding", enabled: true, reason: reason);
        return true;
    }

    private bool ShouldSpeculateForThisIteration()
    {
        // Defensive: if speculation is enabled but we don't have a model forward, we can't speculate.
        return !_speculationDisabledDueToFailure &&
               _model != null &&
               _config.EnableSpeculativeDecoding &&
               _config.SpeculationPolicy != AiDotNet.Configuration.SpeculationPolicy.ForceOff;
    }

    private SpeculativeDecoder<T>? EnsureSpeculativeDecoder()
    {
        if (_speculationDisabledDueToFailure)
            return null;

        if (_speculativeDecoder != null)
            return _speculativeDecoder;

        lock (_speculativeLock)
        {
            if (_speculativeDecoder != null)
                return _speculativeDecoder;

            if (_speculationDisabledDueToFailure)
                return null;

            if (_model == null)
                return null;

            int vocabSize;
            try
            {
                vocabSize = DetectVocabSize();
            }
            catch (Exception ex)
            {
                _speculationDisabledDueToFailure = true;
                InferenceDiagnostics.RecordException("Serving.ContinuousBatching", "SpeculativeDecoding", ex, "Vocab size detection failed; disabling speculation.");
                return null;
            }

            if (vocabSize <= 0)
            {
                _speculationDisabledDueToFailure = true;
                InferenceDiagnostics.RecordDecision("Serving.ContinuousBatching", "SpeculativeDecoding", enabled: false, reason: "DisabledDueToFailure(VocabSizeInvalid)");
                return null;
            }

            IDraftModel<T> draft;
            try
            {
                draft = _draftModelOverride ?? new NGramDraftModel<T>(ngramSize: 3, vocabSize: vocabSize, seed: 42);
            }
            catch (Exception ex)
            {
                _speculationDisabledDueToFailure = true;
                InferenceDiagnostics.RecordException("Serving.ContinuousBatching", "SpeculativeDecoding", ex, "Draft model init failed; disabling speculation.");
                return null;
            }

            Matrix<T> TargetForward(Vector<int> tokens)
            {
                // Run the target model over the full sequence and return per-position probabilities.
                var input = CreateInputTensor(tokens.ToArray());
                var logits = _model(input);

                int seqLen = logits.Shape.Length > 2 ? logits.Shape[^2] : 1;
                int localVocabSize = logits.Shape[^1];

                var numOps = MathHelper.GetNumericOperations<T>();
                var probs = new Matrix<T>(seqLen, localVocabSize);
                for (int pos = 0; pos < seqLen; pos++)
                {
                    // Extract logits for this position
                    var row = new double[localVocabSize];
                    double max = double.NegativeInfinity;
                    for (int v = 0; v < localVocabSize; v++)
                    {
                        double val = Convert.ToDouble(logits[logits.Shape.Length > 2 ? new[] { 0, pos, v } : new[] { 0, v }]);
                        row[v] = val;
                        if (val > max) max = val;
                    }

                    // Softmax
                    double sum = 0.0;
                    for (int v = 0; v < localVocabSize; v++)
                    {
                        row[v] = Math.Exp(row[v] - max);
                        sum += row[v];
                    }
                    if (sum <= 0) sum = 1;

                    for (int v = 0; v < localVocabSize; v++)
                    {
                        probs[pos, v] = numOps.FromDouble(row[v] / sum);
                    }
                }

                return probs;
            }

            var config = new SpeculativeDecodingConfig<T>
            {
                NumDraftTokens = Math.Max(1, _config.SpeculationDepth),
                Seed = 42,
                AdaptiveDraftLength = _config.SpeculationPolicy == AiDotNet.Configuration.SpeculationPolicy.Auto,
                MinAcceptanceRate = MathHelper.GetNumericOperations<T>().FromDouble(0.5),
                UseTreeSpeculation = _config.UseTreeSpeculation ||
                                    _config.SpeculativeMethod == AiDotNet.Configuration.SpeculativeMethod.Medusa ||
                                    _config.SpeculativeMethod == AiDotNet.Configuration.SpeculativeMethod.Eagle,
                TreeBranchFactor = _config.SpeculativeMethod == AiDotNet.Configuration.SpeculativeMethod.Medusa ? 4 : 2,
                MaxTreeDepth = Math.Max(1, _config.SpeculationDepth)
            };

            try
            {
                _speculativeDecoder = new SpeculativeDecoder<T>(draft, TargetForward, config);
                InferenceDiagnostics.RecordDecision("Serving.ContinuousBatching", "SpeculativeDecoding", enabled: true, reason: "DecoderInitialized");
                return _speculativeDecoder;
            }
            catch (Exception ex)
            {
                _speculationDisabledDueToFailure = true;
                InferenceDiagnostics.RecordException("Serving.ContinuousBatching", "SpeculativeDecoding", ex, "Decoder init failed; disabling speculation.");
                return null;
            }
        }
    }

    private int DetectVocabSize()
    {
        try
        {
            // Probe the model with a minimal input to infer the vocabulary dimension.
            var probe = CreateInputTensor([0]);
            var logits = _model!(probe);
            return logits.Shape.Length >= 1 ? logits.Shape[^1] : 0;
        }
        catch
        {
            // Let the caller handle vocab detection failure.
            return 0;
        }
    }

    private Tensor<T> CreateInputTensor(int[] tokenIds)
    {
        // Create a simple 2D tensor [batch=1, seq_len]
        var tensor = new Tensor<T>([1, tokenIds.Length]);
        for (int i = 0; i < tokenIds.Length; i++)
        {
            tensor[[0, i]] = ConvertToT(tokenIds[i]);
        }
        return tensor;
    }

    private T ConvertToT(int value)
    {
        return MathHelper.GetNumericOperations<T>().FromDouble(value);
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
                ? [0, lastPos, i]
                : [0, i];
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
        var random = RandomHelper.ThreadSafeRandom;
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

    private static void ApplyTopP(float[] probs, float topP)
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

    private static void ApplyTopK(float[] probs, int topK)
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
