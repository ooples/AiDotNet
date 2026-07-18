using System.Collections.Concurrent;
using AiDotNet.Helpers;
using AiDotNet.Inference;
using AiDotNet.Inference.PagedAttention;
using AiDotNet.Inference.SpeculativeDecoding;
using AiDotNet.NeuralNetworks;
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

    // Paged incremental-decode path (preferred). When set, each sequence gets an isolated cache sequence in
    // the SHARED PagedKVCache and is decoded incrementally via the model's ForwardWithContext — O(1) per step
    // with real KV caching — instead of the stateless full-context recompute above. Falls back to _model when
    // a model cannot build the paged path.
    private readonly NeuralNetworkBase<T>? _incrementalModel;
    private readonly PagedKVCache<T>? _pagedCache;
    // RadixAttention prompt-prefix sharing over the shared paged cache (built with the paged path). New
    // prompts fork the longest registered strict prefix copy-on-write and forward only the suffix.
    private readonly RadixPrefixCache<T>? _prefixCache;
    // Number of tokens whose KV is already cached for each active sequence (the next ForwardWithContext
    // position), keyed by SequenceState.SequenceId.
    private readonly ConcurrentDictionary<long, int> _pagedPositions = new();

    // Per-sequence RNG so sampling is reproducible per request and concurrent sequences don't share RNG
    // state (the shared ThreadSafeRandom made batched sampling non-deterministic).
    private readonly ConcurrentDictionary<long, Random> _sequenceRngs = new();

    // Draft-model-free PROMPT-LOOKUP speculation accounting for the paged path (n-gram match against the
    // running token stream; exact for greedy decode). Kept separate from the SpeculativeDecoder<T> path.
    private long _pagedDraftedTokens;
    private long _pagedAcceptedTokens;

    private bool UsePagedPath => _incrementalModel is not null && _pagedCache is not null;

    private SpeculativeDecoder<T>? _speculativeDecoder;
    private readonly object _speculativeLock = new();
    private volatile bool _speculationDisabledDueToFailure;
    private long _speculationDisabledUntilIteration;

    internal bool LastStepUsedSpeculation { get; private set; }
    internal int LastStepSpeculationTokens { get; private set; }
    internal string LastStepSpeculationReason { get; private set; } = string.Empty;

    // Number of leading prompt tokens whose KV was reused from a registered RadixAttention prefix on the
    // most recent paged prefill (0 => no prefix reused). Exposed for prefix-sharing tests/telemetry.
    internal int LastPrefillReusedPrefixTokens { get; private set; }

    // Number of sequences served together in the most recent batched paged decode step (1 or 0 => no
    // batched decode ran this step). Exposed for continuous-batching tests/telemetry.
    internal int LastBatchedDecodeCount { get; private set; }

    /// <summary>
    /// Fraction of drafted tokens accepted by the target model, or null when speculative
    /// decoding has not run for this batcher. Exposed for serving-layer telemetry. Covers both the
    /// draft-model path (<see cref="SpeculativeDecoder{T}"/>) and the paged prompt-lookup path.
    /// </summary>
    internal double? SpeculationAcceptanceRate
    {
        get
        {
            if (_pagedDraftedTokens > 0)
            {
                return (double)_pagedAcceptedTokens / _pagedDraftedTokens;
            }
            return _speculativeDecoder?.AcceptanceRate;
        }
    }

    private readonly ConcurrentDictionary<long, TaskCompletionSource<GenerationResult<T>>> _pendingResults;
    private readonly ConcurrentQueue<SequenceState<T>> _incomingRequests;

    // Signals the background run loop that a new request has arrived, so it wakes IMMEDIATELY instead of
    // waiting out its idle poll interval — the difference between ~IdleSleepMs and ~0 first-token latency
    // for a request that arrives while the loop is idle.
    private readonly SemaphoreSlim _workAvailable = new(0);

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
    /// Creates a continuous batcher that decodes each sequence incrementally against a SHARED
    /// <see cref="PagedKVCache{T}"/> (paged KV, O(1) per step) using the model's
    /// <see cref="NeuralNetworkBase{T}.PredictWithContext"/>. This is the fast path — concurrent sequences
    /// share one optimized model and one paged cache, isolated by sequence id. Use this whenever the model
    /// supports the paged incremental path; otherwise use the stateless <see cref="Func{T,TResult}"/> ctor.
    /// </summary>
    /// <param name="config">Batcher configuration.</param>
    /// <param name="incrementalModel">The optimized, context-aware model (writes/reads KV per sequence id).</param>
    /// <param name="pagedCache">The shared paged KV cache backing all sequences.</param>
    /// <param name="draftModel">Optional draft model for speculative decoding.</param>
    public ContinuousBatcher(
        ContinuousBatcherConfig config,
        NeuralNetworkBase<T> incrementalModel,
        PagedKVCache<T> pagedCache,
        IDraftModel<T>? draftModel = null)
    {
        Guard.NotNull(config);
        Guard.NotNull(incrementalModel);
        Guard.NotNull(pagedCache);
        _config = config;
        _incrementalModel = incrementalModel;
        _pagedCache = pagedCache;
        _prefixCache = new RadixPrefixCache<T>(pagedCache);
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
        // Wake the run loop now if it is idle-waiting (see _workAvailable). Safe to over-signal: extra
        // permits just cause a spurious no-op Step before the loop settles back to waiting.
        try { _workAvailable.Release(); } catch (System.ObjectDisposedException) { }

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

        // Partition the batch: sequences still needing prefill vs. those already decoding.
        // decodeSequences is captured BEFORE prefill flips PrefillComplete, so a sequence
        // prefilled this iteration emits exactly its prefill token now and only starts
        // decoding on subsequent iterations. This honors MaxNewTokens exactly (a fresh
        // sequence no longer both prefills and decodes within the same step).
        var prefillSequences = batch.Where(s => !s.PrefillComplete).ToList();
        var decodeSequences = batch.Where(s => s.PrefillComplete).ToList();

        // Run prefill for new sequences and account for the first generated token. On the paged path with
        // more than one new (non-empty) sequence, prefill them ALL in ONE right-padded batched forward
        // (Phase 2); fall back to per-sequence prefill otherwise or if the batched attempt can't allocate.
        // Batched prefill pads prompts to a common length; a sequence-collapsing model (Flatten head,
        // SupportsBatchedPrefill=false) would see the padded width and diverge from a single-prompt
        // forward, so only batch prefill for models that accept a multi-token forward.
        bool batchedPrefillDone = false;
        if (UsePagedPath && _config.SupportsBatchedPrefill && _config.MaxPrefillChunkTokens <= 0 && prefillSequences.Count > 1)
        {
            var batchedPrefill = RunBatchedPagedPrefill(prefillSequences);
            if (batchedPrefill is not null)
            {
                foreach (var (seq, token) in batchedPrefill)
                {
                    tokensGenerated += AccountForToken(seq, token, useSpeculation);
                }
                batchedPrefillDone = true;
            }
        }
        if (!batchedPrefillDone)
        {
            foreach (var seq in prefillSequences)
            {
                int? firstToken = RunPrefill(seq);
                if (firstToken.HasValue)
                {
                    tokensGenerated += AccountForToken(seq, firstToken.Value, useSpeculation);
                }
            }
        }

        // Run a decode step for sequences that were already generating in prior iterations. When the paged
        // path is active, speculation is off, and more than one sequence is decoding, serve them ALL in ONE
        // batched forward (true continuous batching) instead of one forward per sequence — the Phase 2
        // throughput win. Otherwise decode per sequence (speculation, single-sequence, or stateless path).
        var generating = decodeSequences.Where(s => s.Status == SequenceStatus.Generating).ToList();
        if (UsePagedPath && generating.Count > 1)
        {
            // One batched forward serves the whole decode batch: plain batched decode when speculation is
            // off, and batched speculative verify (grouped by draft length) when it is on.
            foreach (var (seq, tokens) in RunPagedDecodeBatch(generating, useSpeculation))
            {
                foreach (var token in tokens)
                {
                    tokensGenerated += AccountForToken(seq, token, useSpeculation);
                    if (seq.Status == SequenceStatus.Completed) break;
                }
            }
        }
        else
        {
            foreach (var seq in decodeSequences)
            {
                if (seq.Status != SequenceStatus.Generating)
                    continue;

                var newTokens = useSpeculation ? RunDecodeStepSpeculative(seq) : RunDecodeStep(seq);
                foreach (var newToken in newTokens)
                {
                    tokensGenerated += AccountForToken(seq, newToken, useSpeculation);

                    // Stop processing further tokens once the sequence has completed.
                    if (seq.Status == SequenceStatus.Completed)
                        break;
                }
            }
        }

        return tokensGenerated;
    }

    /// <summary>
    /// Records a generated token: updates statistics, raises the streaming notification and
    /// per-token callback, and completes the sequence when a stop condition is reached.
    /// The token must already be appended to the sequence by the prefill/decode step that
    /// produced it.
    /// </summary>
    /// <returns>The number of tokens accounted for (always 1).</returns>
    private int AccountForToken(SequenceState<T> sequence, int tokenId, bool useSpeculation)
    {
        _totalTokensGenerated++;
        if (useSpeculation)
            LastStepSpeculationTokens++;

        // Fire token-generated event (for streaming consumers).
        TokenGenerated?.Invoke(this, new TokenGeneratedEventArgs<T>
        {
            Sequence = sequence,
            TokenId = tokenId
        });

        // Invoke the per-token callback if one was provided.
        sequence.Request.OnTokenGenerated?.Invoke(tokenId);

        // Check for completion after the appended token. EOS is per-request (a shared batcher serves
        // requests with different stop tokens), falling back to the batcher default.
        if (sequence.Status != SequenceStatus.Completed &&
            sequence.ShouldStop(sequence.Request.EosTokenId ?? _config.EosTokenId, sequence.Request.StopTokenIds))
        {
            CompleteSequence(sequence);
        }

        return 1;
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

                // Nothing to do: block until a new request signals _workAvailable (immediate wake) or the
                // idle interval elapses (fallback re-check). This avoids busy-polling AND the up-to-
                // IdleSleepMs first-token latency a plain Task.Delay would add for a request that arrives
                // right after the loop goes idle.
                if (tokensGenerated == 0 && _scheduler.WaitingCount == 0 && _scheduler.RunningCount == 0
                    && _incomingRequests.IsEmpty)
                {
                    await _workAvailable.WaitAsync(_config.IdleSleepMs, cancellationToken).ConfigureAwait(false);
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

    /// <summary>
    /// Processes the full prompt in a single forward pass and emits the first generated token.
    /// The token (if any) is appended to the sequence and returned so the caller can run the
    /// shared per-token accounting; returns null when there is no model or no prompt tokens.
    /// </summary>
    private int? RunPrefill(SequenceState<T> sequence)
    {
        sequence.Status = SequenceStatus.Prefilling;
        sequence.GenerationStartedAt ??= DateTime.UtcNow;

        int? firstToken = null;
        bool complete = true;
        if (UsePagedPath && sequence.TokenIds.Count > 0)
        {
            firstToken = RunPagedPrefill(sequence);
            // Chunked prefill: complete only once the cursor has consumed the whole prompt. A partial
            // chunk (or an allocation retry) leaves the sequence Prefilling for the next step.
            complete = sequence.PrefillCursor >= sequence.PromptLength;
        }
        else if (_model != null && sequence.TokenIds.Count > 0)
        {
            // Stateless full-context prefill (fallback for models without the paged incremental path).
            var inputTokens = CreateInputTensor(sequence.TokenIds);
            var logits = _model(inputTokens);
            int nextToken = SampleFromLogits(logits, sequence);
            sequence.AppendToken(nextToken);
            firstToken = nextToken;
        }

        if (complete)
        {
            sequence.PrefillComplete = true;
            sequence.Status = SequenceStatus.Generating;
        }
        return firstToken;
    }

    // Paged prefill (chunk-aware): write the next CHUNK of the prompt's KV into this sequence's paged cache,
    // advancing PrefillCursor. When the cursor reaches the prompt end the prompt is fully cached, so the
    // first token is sampled and returned; otherwise null is returned and the sequence stays Prefilling for
    // its next chunk (interleaving a long prefill with ongoing decode). Chunking (MaxPrefillChunkTokens>0)
    // applies to the batched-prefill path; sequence-collapsing models prefill per-token in one step.
    private int? RunPagedPrefill(SequenceState<T> sequence)
    {
        if (_pagedCache is not { } cache || _incrementalModel is not { } model) return null;

        long seqId = sequence.SequenceId;
        var prompt = sequence.TokenIds;
        int promptLen = prompt.Count;

        // First chunk: allocate the cache slot and reuse the longest registered STRICT prefix
        // (copy-on-write), which sets the starting cursor. cached < promptLen always.
        if (sequence.PrefillCursor == 0)
        {
            if (!cache.AllocateSequence(seqId, 0)) return null; // cache exhausted / id in use — retry next step
            int cached = _prefixCache is { } prefixCache ? prefixCache.TryForkLongestPrefix(prompt, seqId) : 0;
            if (cached >= promptLen) cached = promptLen - 1;
            LastPrefillReusedPrefixTokens = cached;
            sequence.PrefillCursor = cached;
        }

        int start = sequence.PrefillCursor;

        // Forward the next piece of the prompt from `start`. Batched-prefill models take a multi-token
        // chunk in one pass (verified equivalent to per-token); sequence-collapsing (Flatten) models must
        // go one token at a time (a multi-token forward re-fits a shape-dependent head) and aren't chunked.
        Tensor<T> logits;
        if (_config.SupportsBatchedPrefill)
        {
            int remaining = promptLen - start;
            int chunk = _config.MaxPrefillChunkTokens > 0 ? Math.Min(remaining, _config.MaxPrefillChunkTokens) : remaining;
            var piece = new List<int>(chunk);
            for (int i = 0; i < chunk; i++) piece.Add(prompt[start + i]);
            logits = model.PredictWithContext(CreateInputTensor(piece), new InferenceForwardContext(seqId, start));
            sequence.PrefillCursor = start + chunk;
        }
        else
        {
            logits = model.PredictWithContext(
                CreateInputTensor(new List<int> { prompt[start] }), new InferenceForwardContext(seqId, start));
            for (int i = start + 1; i < promptLen; i++)
            {
                logits = model.PredictWithContext(
                    CreateInputTensor(new List<int> { prompt[i] }), new InferenceForwardContext(seqId, i));
            }
            sequence.PrefillCursor = promptLen;
        }
        _pagedPositions[seqId] = sequence.PrefillCursor;

        // More chunks remain: emit no token this step; the sequence stays Prefilling.
        if (sequence.PrefillCursor < promptLen) return null;

        // Prefill complete: register the prompt as a reusable prefix (KV now holds exactly the prompt),
        // then sample the first token.
        _prefixCache?.Register(prompt, seqId);
        int nextToken = SampleFromLogits(logits, sequence);
        sequence.AppendToken(nextToken);
        return nextToken;
    }

    /// <summary>
    /// Batched paged prefill (Phase 2): prefill MULTIPLE new sequences in ONE right-padded forward
    /// (<c>[batch, maxPromptLen]</c> input; per-row lengths mask the padded tail), then sample each
    /// sequence's first token from its own last real position. Prefix reuse is skipped for the batched
    /// group (whole prompts are forwarded), which keeps positions uniform (all start at 0). Returns null
    /// when it cannot batch (a cache allocation failed) so the caller falls back to per-sequence prefill.
    /// Per-row output is identical to a single-sequence prefill (verified).
    /// </summary>
    private List<(SequenceState<T> Sequence, int Token)>? RunBatchedPagedPrefill(List<SequenceState<T>> sequences)
    {
        if (_pagedCache is not { } cache || _incrementalModel is not { } model) return null;
        int batch = sequences.Count;
        if (batch == 0) return null;

        // Allocate a fresh cache sequence per row; on any failure, roll back and signal a per-seq fallback.
        var seqIds = new long[batch];
        var promptLens = new int[batch];
        int maxLen = 0;
        for (int i = 0; i < batch; i++)
        {
            int promptLen = sequences[i].TokenIds.Count;
            if (promptLen <= 0) { for (int j = 0; j < i; j++) cache.FreeSequence(seqIds[j]); return null; }
            long seqId = sequences[i].SequenceId;
            if (!cache.AllocateSequence(seqId, 0))
            {
                for (int j = 0; j < i; j++) cache.FreeSequence(seqIds[j]);
                return null;
            }
            seqIds[i] = seqId;
            promptLens[i] = promptLen;
            if (promptLen > maxLen) maxLen = promptLen;
        }

        var positions = new int[batch]; // fresh prefill: every row starts at position 0
        var input = new Tensor<T>([batch, maxLen]);
        for (int i = 0; i < batch; i++)
        {
            var seq = sequences[i];
            seq.Status = SequenceStatus.Prefilling;
            seq.GenerationStartedAt = DateTime.UtcNow;
            var toks = seq.TokenIds;
            for (int t = 0; t < promptLens[i]; t++) input[[i, t]] = ConvertToT(toks[t]);
            // Right-padded tail (t >= promptLens[i]) stays 0 and is masked out by RowLengths.
        }

        var ctx = new InferenceForwardContext(seqIds, positions) { RowLengths = promptLens };
        var logits = model.PredictWithContext(input, ctx);

        var results = new List<(SequenceState<T>, int)>(batch);
        for (int i = 0; i < batch; i++)
        {
            var seq = sequences[i];
            _pagedPositions[seq.SequenceId] = promptLens[i];
            // Register the prompt (TokenIds still holds exactly the prompt) as a reusable prefix.
            _prefixCache?.Register(seq.TokenIds, seq.SequenceId);
            int nextToken = SampleFromLogits(logits, seq, batchIndex: i, lastPositionOverride: promptLens[i] - 1);
            seq.AppendToken(nextToken);
            seq.PrefillComplete = true;
            seq.Status = SequenceStatus.Generating;
            results.Add((seq, nextToken));
        }
        return results;
    }

    private IReadOnlyList<int> RunDecodeStep(SequenceState<T> sequence)
    {
        if (UsePagedPath) return RunPagedDecodeStep(sequence);
        if (_model == null) return Array.Empty<int>();

        // Forward the full running sequence (prompt + tokens generated so far) so the model attends
        // to the complete context. The model forward (_model) is stateless — it does not retain KV
        // state across calls — so a last-token-only forward would discard all prior context and
        // produce incorrect logits for any context-dependent model (e.g. a transformer LM). This
        // matches the speculative path's TargetForward, which already forwards the full sequence.
        // (A KV-cached incremental forward is a future optimization that would require a stateful
        // model-forward contract; full-context decode is the correct behavior for a stateless one.)
        var inputTokens = CreateInputTensor(sequence.TokenIds);

        // Run model forward pass
        var logits = _model(inputTokens);

        // Sample next token
        int nextToken = SampleFromLogits(logits, sequence);
        sequence.AppendToken(nextToken);

        return new[] { nextToken };
    }

    // Paged decode: forward ONLY the last token against the cached KV for this sequence (O(1) per step).
    private IReadOnlyList<int> RunPagedDecodeStep(SequenceState<T> sequence)
    {
        if (_incrementalModel is not { } model) return Array.Empty<int>();

        long seqId = sequence.SequenceId;
        int position = _pagedPositions.TryGetValue(seqId, out var p) ? p : sequence.PromptLength;
        int lastToken = sequence.TokenIds[sequence.TokenIds.Count - 1];

        var input = CreateInputTensor(new List<int> { lastToken });
        var logits = model.PredictWithContext(input, new InferenceForwardContext(seqId, position));
        _pagedPositions[seqId] = position + 1;

        int nextToken = SampleFromLogits(logits, sequence);
        sequence.AppendToken(nextToken);
        return new[] { nextToken };
    }

    /// <summary>
    /// Batched paged decode (Phase 2 — true continuous batching): forward the last token of EVERY given
    /// sequence in ONE batched pass ([batch, 1] input, per-row sequence id + position), then sample each
    /// sequence's next token from its own row. One forward serves the whole decode batch instead of one
    /// forward per sequence. Per-row output is byte-identical to a single-sequence decode (verified).
    /// </summary>
    private List<(SequenceState<T> Sequence, int Token)> RunBatchedPagedDecodeStep(List<SequenceState<T>> sequences)
    {
        var results = new List<(SequenceState<T>, int)>(sequences.Count);
        if (_incrementalModel is not { } model || sequences.Count == 0) return results;

        int batch = sequences.Count;
        LastBatchedDecodeCount = batch;
        var seqIds = new long[batch];
        var positions = new int[batch];
        var input = new Tensor<T>([batch, 1]);
        for (int i = 0; i < batch; i++)
        {
            var seq = sequences[i];
            long seqId = seq.SequenceId;
            int position = _pagedPositions.TryGetValue(seqId, out var p) ? p : seq.PromptLength;
            seqIds[i] = seqId;
            positions[i] = position;
            input[[i, 0]] = ConvertToT(seq.TokenIds[seq.TokenIds.Count - 1]);
        }

        var logits = model.PredictWithContext(input, new InferenceForwardContext(seqIds, positions));

        for (int i = 0; i < batch; i++)
        {
            var seq = sequences[i];
            int nextToken = SampleFromLogits(logits, seq, batchIndex: i);
            seq.AppendToken(nextToken);
            _pagedPositions[seq.SequenceId] = positions[i] + 1;
            results.Add((seq, nextToken));
        }
        return results;
    }

    /// <summary>
    /// Greedy-exact prompt-lookup speculative decode over the paged cache. Forwards the last token to get
    /// the target's greedy next token, drafts up to SpeculationDepth tokens by matching the trailing n-gram
    /// against the running stream, verifies them in ONE multi-token forward (per-position logits), accepts
    /// the longest greedy-matching prefix, rolls the paged KV back over rejected drafts, and emits the
    /// correction. Emitted tokens are byte-for-byte identical to plain greedy decode — speculation only
    /// changes how many forwards it takes. Falls back to a single sampled step for non-greedy requests
    /// (prompt-lookup is exact only for greedy) or when no n-gram match exists.
    /// </summary>
    private IReadOnlyList<int> RunPagedDecodeStepSpeculative(SequenceState<T> sequence)
    {
        // Prompt-lookup speculation is exact only for greedy decode; sample normally otherwise.
        if (sequence.Request.Temperature > 0f) return RunPagedDecodeStep(sequence);
        if (_incrementalModel is not { } model) return Array.Empty<int>();

        // Speculation verifies K drafts in ONE multi-token forward, so it needs a model that yields
        // per-position logits. Sequence-collapsing models cannot; decode them one token at a time.
        if (!_config.SupportsBatchedPrefill) return RunPagedDecodeStep(sequence);

        // Per-request draft depth (null => batcher default). 0 disables speculation for this request.
        int depth = sequence.Request.SpeculationDepth ?? _config.SpeculationDepth;
        if (depth <= 0) return RunPagedDecodeStep(sequence);

        long seqId = sequence.SequenceId;
        int remaining = sequence.MaxNewTokens - sequence.GeneratedLength;
        if (remaining <= 0) return Array.Empty<int>();

        int eos = sequence.Request.EosTokenId ?? _config.EosTokenId;
        int p = _pagedPositions.TryGetValue(seqId, out var pos) ? pos : sequence.PromptLength;
        int lastToken = sequence.TokenIds[sequence.TokenIds.Count - 1];

        // 1) Forward the last token -> greedy next token (KV[p] now written; cache length p+1).
        var greedyLogits = model.PredictWithContext(
            CreateInputTensor(new List<int> { lastToken }), new InferenceForwardContext(seqId, p));
        int greedy = ArgMaxLastPosition(greedyLogits);

        // 2) Draft by matching the trailing n-gram against the running stream.
        int k = Math.Max(1, depth);
        var draft = PromptLookupDraft(sequence.TokenIds, k);

        var emitted = new List<int>(k + 1);

        if (draft.Count == 0)
        {
            // No match: a single ordinary greedy step. greedy's KV is written when the NEXT step forwards it.
            _pagedPositions[seqId] = p + 1;
            emitted.Add(greedy);
            sequence.AppendToken(greedy);
            return emitted;
        }

        // 3) Verify the K drafts in ONE forward at position p+1 (KV[p+1..p+K]; cache length p+1+K).
        var verifyLogits = model.PredictWithContext(
            CreateInputTensor(draft), new InferenceForwardContext(seqId, p + 1));
        _pagedDraftedTokens += draft.Count;

        // 4-6) Accept the longest greedy-matching prefix, roll back rejected KV, emit the correction.
        return EmitSpeculativeRow(sequence, greedy, draft, p, verifyLogits, batchRow: 0);
    }

    /// <summary>
    /// Steps 4-6 of greedy-exact speculative decode for ONE sequence, shared by the per-sequence and
    /// batched paths: accept the longest prefix where draft[j] == the target's greedy token
    /// (expected_0 = <paramref name="greedy"/>, expected_{j+1} = argmax of verify row j), emit the accepted
    /// drafts (capped by remaining budget / EOS), roll the paged KV back over the rejected drafts
    /// (<see cref="PagedKVCache{T}.TruncateSequence"/>), and emit the correction. <paramref name="batchRow"/>
    /// selects this sequence's row in a batched verify tensor (0 for a single-sequence forward).
    /// </summary>
    private List<int> EmitSpeculativeRow(
        SequenceState<T> sequence, int greedy, List<int> draft, int p, Tensor<T> verifyLogits, int batchRow)
    {
        long seqId = sequence.SequenceId;
        int eos = sequence.Request.EosTokenId ?? _config.EosTokenId;
        int remaining = sequence.MaxNewTokens - sequence.GeneratedLength;
        var emitted = new List<int>(draft.Count + 1);

        int expected = greedy;
        int nAccept = 0;
        for (int j = 0; j < draft.Count; j++)
        {
            if (draft[j] != expected) break;
            nAccept++;
            expected = ArgMaxAtBatchPosition(verifyLogits, batchRow, j);
        }
        _pagedAcceptedTokens += nAccept;

        for (int j = 0; j < nAccept; j++)
        {
            if (draft[j] == eos) { FinalizePagedSpeculation(seqId, p + 1 + j, sequence, emitted, draft[j]); return emitted; }
            emitted.Add(draft[j]);
            sequence.AppendToken(draft[j]);
            if (emitted.Count >= remaining)
            {
                // Budget reached: the last emitted draft's KV (p+1+j) is real; next write is p+2+j.
                _pagedPositions[seqId] = p + 2 + j;
                return emitted;
            }
        }

        _pagedCache?.TruncateSequence(seqId, p + 1 + nAccept);
        _pagedPositions[seqId] = p + 1 + nAccept;
        if (expected != eos)
        {
            emitted.Add(expected);
            sequence.AppendToken(expected);
        }
        return emitted;
    }

    /// <summary>
    /// Batched greedy-exact speculative decode for a group of sequences that all drafted the SAME number of
    /// tokens K &gt; 0: forward every sequence's last token in ONE batched pass (greedy), verify all K-token
    /// drafts in ONE batched pass, then per-row accept/roll-back/emit. Byte-identical to per-sequence
    /// speculation, which is itself identical to plain greedy — batching only changes the forward count.
    /// </summary>
    private List<(SequenceState<T> Sequence, IReadOnlyList<int> Tokens)> RunBatchedPagedSpeculativeStep(
        List<SequenceState<T>> group, List<List<int>> drafts)
    {
        var results = new List<(SequenceState<T>, IReadOnlyList<int>)>(group.Count);
        if (_incrementalModel is not { } model || group.Count == 0) return results;

        int b = group.Count;
        int k = drafts[0].Count;
        var seqIds = new long[b];
        var greedyPos = new int[b];
        var greedyInput = new Tensor<T>([b, 1]);
        for (int i = 0; i < b; i++)
        {
            var seq = group[i];
            long seqId = seq.SequenceId;
            int p = _pagedPositions.TryGetValue(seqId, out var pp) ? pp : seq.PromptLength;
            seqIds[i] = seqId;
            greedyPos[i] = p;
            greedyInput[[i, 0]] = ConvertToT(seq.TokenIds[seq.TokenIds.Count - 1]);
        }

        // 1) Batched greedy forward (last token of each @ its own position).
        var greedyLogits = model.PredictWithContext(greedyInput, new InferenceForwardContext(seqIds, greedyPos));

        // 2) Batched verify forward (K drafts of each @ position+1).
        var verifyPos = new int[b];
        var verifyInput = new Tensor<T>([b, k]);
        for (int i = 0; i < b; i++)
        {
            verifyPos[i] = greedyPos[i] + 1;
            for (int j = 0; j < k; j++) verifyInput[[i, j]] = ConvertToT(drafts[i][j]);
        }
        var verifyLogits = model.PredictWithContext(verifyInput, new InferenceForwardContext(seqIds, verifyPos));
        _pagedDraftedTokens += (long)b * k;

        // 3) Per-row accept + emit.
        for (int i = 0; i < b; i++)
        {
            int greedy = ArgMaxAtBatchPosition(greedyLogits, i, 0); // [b,1,vocab]: row i, position 0
            var emitted = EmitSpeculativeRow(group[i], greedy, drafts[i], greedyPos[i], verifyLogits, batchRow: i);
            results.Add((group[i], emitted));
        }
        return results;
    }

    /// <summary>
    /// Multi-sequence paged decode step. Without speculation, one batched decode forward serves all
    /// sequences. With speculation, sequences are drafted (prompt-lookup) and partitioned: those with no
    /// draft decode via one batched forward, and those with a draft are grouped by draft length and each
    /// same-length group is verified in one batched speculative forward (singletons decode per-sequence).
    /// </summary>
    private List<(SequenceState<T> Sequence, IReadOnlyList<int> Tokens)> RunPagedDecodeBatch(
        List<SequenceState<T>> generating, bool useSpeculation)
    {
        var results = new List<(SequenceState<T>, IReadOnlyList<int>)>(generating.Count);

        if (!useSpeculation)
        {
            foreach (var (seq, token) in RunBatchedPagedDecodeStep(generating)) results.Add((seq, new[] { token }));
            return results;
        }

        // Speculation on: draft each eligible sequence; batch no-draft/ineligible ones as plain decode, and
        // group drafted ones by draft length for a batched speculative verify.
        var plain = new List<SequenceState<T>>();
        var groups = new Dictionary<int, (List<SequenceState<T>> Seqs, List<List<int>> Drafts)>();
        foreach (var seq in generating)
        {
            int depth = seq.Request.SpeculationDepth ?? _config.SpeculationDepth;
            bool eligible = seq.Request.Temperature <= 0f && _config.SupportsBatchedPrefill && depth > 0
                            && (seq.MaxNewTokens - seq.GeneratedLength) > 0;
            if (!eligible) { plain.Add(seq); continue; }
            var draft = PromptLookupDraft(seq.TokenIds, Math.Max(1, depth));
            if (draft.Count == 0) { plain.Add(seq); continue; }
            if (!groups.TryGetValue(draft.Count, out var g)) { g = (new List<SequenceState<T>>(), new List<List<int>>()); groups[draft.Count] = g; }
            g.Seqs.Add(seq);
            g.Drafts.Add(draft);
        }

        if (plain.Count > 1)
        {
            foreach (var (seq, token) in RunBatchedPagedDecodeStep(plain)) results.Add((seq, new[] { token }));
        }
        else if (plain.Count == 1)
        {
            results.Add((plain[0], RunPagedDecodeStep(plain[0])));
        }

        foreach (var kv in groups)
        {
            var g = kv.Value;
            if (g.Seqs.Count > 1) results.AddRange(RunBatchedPagedSpeculativeStep(g.Seqs, g.Drafts));
            else results.Add((g.Seqs[0], RunPagedDecodeStepSpeculative(g.Seqs[0])));
        }
        return results;
    }

    // Appends the EOS token and returns; its KV position is irrelevant (generation stops here).
    private void FinalizePagedSpeculation(long seqId, int writePosition, SequenceState<T> sequence, List<int> emitted, int eosToken)
    {
        _pagedPositions[seqId] = writePosition;
        emitted.Add(eosToken);
        sequence.AppendToken(eosToken);
    }

    /// <summary>
    /// Prompt-lookup draft: find the most recent earlier occurrence of the trailing <paramref name="ngram"/>
    /// tokens in the running stream and propose the up-to-<paramref name="k"/> tokens that followed it. Empty
    /// when no match exists. No draft model needed.
    /// </summary>
    private static List<int> PromptLookupDraft(IReadOnlyList<int> tokens, int k, int ngram = 2)
    {
        int n = tokens.Count;
        var draft = new List<int>(k);
        if (n < ngram + 1) return draft;

        for (int start = n - ngram - 1; start >= 0; start--)
        {
            bool match = true;
            for (int g = 0; g < ngram; g++)
            {
                if (tokens[start + g] != tokens[n - ngram + g]) { match = false; break; }
            }
            if (!match) continue;

            int src = start + ngram;
            for (int i = 0; i < k && src + i < n; i++) draft.Add(tokens[src + i]);
            return draft;
        }
        return draft;
    }

    // Argmax over the last position of a logits tensor ([1,S,vocab], [S,vocab], [1,vocab], or [vocab]).
    private static int ArgMaxLastPosition(Tensor<T> logits)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int rank = logits.Shape.Length;
        int vocab = logits.Shape[rank - 1];
        int positions = 1;
        for (int d = 0; d < rank - 1; d++) positions *= logits.Shape[d];
        int baseOffset = (positions - 1) * vocab;
        var flat = logits.AsSpan();
        int best = 0;
        T bestVal = flat[baseOffset];
        for (int v = 1; v < vocab; v++)
        {
            if (numOps.GreaterThan(flat[baseOffset + v], bestVal)) { bestVal = flat[baseOffset + v]; best = v; }
        }
        return best;
    }

    // Argmax over vocab at (batch row, output position) of a per-position logits tensor
    // ([batch, positions, vocab]; batchRow 0 and rank-2 [positions, vocab] also handled).
    private static int ArgMaxAtBatchPosition(Tensor<T> logits, int batchRow, int posIndex)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int rank = logits.Shape.Length;
        int vocab = logits.Shape[rank - 1];
        int seq = rank >= 3 ? logits.Shape[rank - 2] : 1;
        int baseOffset = (batchRow * seq + posIndex) * vocab;
        var flat = logits.AsSpan();
        int best = 0;
        T bestVal = flat[baseOffset];
        for (int v = 1; v < vocab; v++)
        {
            if (numOps.GreaterThan(flat[baseOffset + v], bestVal)) { bestVal = flat[baseOffset + v]; best = v; }
        }
        return best;
    }

    private IReadOnlyList<int> RunDecodeStepSpeculative(SequenceState<T> sequence)
    {
        // Paged incremental path: use draft-model-free PROMPT-LOOKUP speculation (n-gram match against the
        // running token stream, verified in one multi-token forward over the paged KV). Exact for greedy.
        if (UsePagedPath) return RunPagedDecodeStepSpeculative(sequence);
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
        // Structured-output requests must never speculate: the greedy draft/verify path picks tokens via
        // argmax without applying the per-sequence constraint mask, so a drafted token could violate the
        // required format. Force plain masked decode whenever any batched sequence carries a constraint.
        // This overrides even ForceOn — correctness of the constraint takes priority over the speed hint.
        foreach (var seq in batch)
        {
            if (seq.Request.Constraint is not null)
            {
                reason = "StructuredOutputConstraint";
                InferenceDiagnostics.RecordDecision("Serving.ContinuousBatching", "SpeculativeDecoding", enabled: false, reason: reason);
                return false;
            }
        }

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
                // Honor explicit config values when provided (> 0); otherwise derive sensible defaults.
                TreeBranchFactor = _config.TreeBranchFactor > 0
                    ? _config.TreeBranchFactor
                    : (_config.SpeculativeMethod == AiDotNet.Configuration.SpeculativeMethod.Medusa ? 4 : 2),
                MaxTreeDepth = _config.MaxTreeDepth > 0
                    ? _config.MaxTreeDepth
                    : Math.Max(1, _config.SpeculationDepth)
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

    private Tensor<T> CreateInputTensor(IReadOnlyList<int> tokenIds)
    {
        // Create a simple 2D tensor [batch=1, seq_len]. Accepts IReadOnlyList so callers can pass
        // the live token list directly — no per-step ToArray() copy in the decode hot path.
        var tensor = new Tensor<T>([1, tokenIds.Count]);
        for (int i = 0; i < tokenIds.Count; i++)
        {
            tensor[[0, i]] = ConvertToT(tokenIds[i]);
        }
        return tensor;
    }

    private T ConvertToT(int value)
    {
        return MathHelper.GetNumericOperations<T>().FromDouble(value);
    }

    private int SampleFromLogits(Tensor<T> logits, SequenceState<T> sequence, int batchIndex = 0, int lastPositionOverride = -1)
    {
        var request = sequence.Request;

        // Get last position logits (shape is typically [batch, seq, vocab]). batchIndex selects the row for
        // a BATCHED forward ([B, seq, vocab] / [B, vocab]); it is 0 for a single-sequence forward.
        // lastPositionOverride (>= 0) picks a specific position instead of the last — used by batched
        // prefill, where each right-padded row's real final token is at rowLength-1, not maxLen-1.
        int vocabSize = logits.Shape[^1];
        int lastPos = logits.Shape.Length > 2
            ? (lastPositionOverride >= 0 ? lastPositionOverride : logits.Shape[^2] - 1)
            : 0;

        // Extract logits for sampling
        var lastLogits = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            int[] indices = logits.Shape.Length > 2
                ? [batchIndex, lastPos, i]
                : [batchIndex, i];
            lastLogits[i] = Convert.ToSingle(logits[indices]);
        }

        // Structured-output constraint (JSON / regex / grammar / choice): forbid tokens that would break
        // the required format BEFORE greedy/softmax/sampling by setting their logit to -inf. Applied here so
        // it composes with temperature, min-p, top-p, and top-k. Speculation is disabled for constrained
        // requests (see ShouldUseSpeculativeDecoding), so every emitted token passes through this mask.
        var constraint = request.Constraint;
        constraint?.ApplyMask(lastLogits);

        // Advances the constraint state for the chosen token and returns it — the single exit point so the
        // constraint sees exactly the committed token regardless of which sampling branch produced it.
        int Finalize(int token)
        {
            constraint?.Accept(token);
            return token;
        }

        // Greedy decoding (temperature <= 0): the argmax — deterministic, and avoids dividing by zero below.
        if (request.Temperature <= 0f)
        {
            int best = 0;
            for (int i = 1; i < vocabSize; i++)
            {
                if (lastLogits[i] > lastLogits[best]) best = i;
            }
            return Finalize(best);
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

        // Apply min-p: drop tokens below minP × the top token's probability, then renormalize.
        if (request.MinP > 0.0f)
        {
            ApplyMinP(lastLogits, request.MinP);
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

        // Sample from the per-sequence deterministic RNG (seeded from the request) so identical seeded
        // requests are reproducible and concurrent sequences don't share RNG state.
        var random = GetSequenceRng(sequence);
        float r = (float)random.NextDouble();
        float cumSum = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            cumSum += lastLogits[i];
            if (cumSum >= r)
                return Finalize(i);
        }

        return Finalize(vocabSize - 1); // Fallback to last token
    }

    // Per-sequence deterministic RNG: seeded from the request when a seed is provided (reproducible),
    // otherwise cryptographically secure. Persisted per sequence so draws advance across decode steps.
    private Random GetSequenceRng(SequenceState<T> sequence)
        => _sequenceRngs.GetOrAdd(sequence.SequenceId, _ =>
            sequence.Request.Seed is { } seed
                ? RandomHelper.CreateSeededRandom(seed)
                : RandomHelper.CreateSecureRandom());

    private static void ApplyMinP(float[] probs, float minP)
    {
        float max = 0f;
        for (int i = 0; i < probs.Length; i++)
        {
            if (probs[i] > max) max = probs[i];
        }
        float threshold = minP * max;
        float sum = 0f;
        for (int i = 0; i < probs.Length; i++)
        {
            if (probs[i] < threshold) probs[i] = 0f;
            sum += probs[i];
        }
        if (sum > 0f)
        {
            for (int i = 0; i < probs.Length; i++) probs[i] /= sum;
        }
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
        // Release this sequence's paged KV blocks (no-op on the stateless path) and its RNG.
        if (_pagedCache is { } cache)
        {
            cache.FreeSequence(sequence.SequenceId);
            _pagedPositions.TryRemove(sequence.SequenceId, out _);
        }
        _sequenceRngs.TryRemove(sequence.SequenceId, out _);

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
        _workAvailable.Dispose();

        GC.SuppressFinalize(this);
    }
}
