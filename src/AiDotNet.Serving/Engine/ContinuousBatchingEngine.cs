using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Engine;

/// <summary>
/// The continuous- (in-flight-) batching inference engine: the loop that keeps many requests progressing at
/// once over a paged KV cache, the way vLLM and TGI do. Requests are admitted any time via
/// <see cref="AddRequest"/>; each <see cref="Step"/> schedules a batch (prefill for freshly admitted prompts,
/// one-token decode for running sequences), runs the model once, samples a token per sequence, frees finished
/// sequences, and — under KV-memory pressure — preempts by recompute so the batch always fits.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> a naive server runs one request at a time. This engine instead interleaves them:
/// every step it advances every in-flight request by one token and slots in new requests the moment memory is
/// free, so the hardware stays busy and everyone's latency stays low. It borrows two ideas from vLLM — a paged
/// KV cache (memory in small reusable blocks, see <see cref="BlockManager"/>) and continuous batching — behind
/// a simple <see cref="IInferenceEngine"/> you just pump in a loop.</para>
/// <para>The engine is model-agnostic: it drives an <see cref="IServingModelRunner{T}"/>, which is satisfied
/// either by a paged fast path (a model's <see cref="ICausalLmRunner{T}"/> capability) or by a recompute
/// fallback over any <see cref="AiDotNet.Interfaces.IFullModel{T, TInput, TOutput}"/>. This keeps the scheduler
/// identical regardless of model, and lets it be tested against a deterministic fake runner.</para>
/// <para>Not thread-safe for concurrent <see cref="Step"/> calls (drive it from one loop); <see cref="AddRequest"/>
/// and <see cref="AbortRequest"/> are safe to call from other threads while stepping.</para>
/// </remarks>
/// <typeparam name="T">The model's numeric type.</typeparam>
public sealed class ContinuousBatchingEngine<T> : IInferenceEngine
{
    private readonly IServingModelRunner<T> _runner;
    private readonly EngineOptions _options;
    private readonly BlockManager _blocks;

    private readonly object _intakeLock = new();
    private readonly LinkedList<Sequence> _waiting = new();
    private readonly List<Sequence> _running = new();            // kept oldest-first (arrival order)
    private readonly Dictionary<string, List<Sequence>> _byRequest = new();
    private readonly Dictionary<string, Random> _rngBySequence = new();
    private readonly Queue<GenerationRequest> _pendingAdds = new();
    private readonly HashSet<string> _pendingAborts = new();
    // For N>1 parallel sampling: siblings that wait to be forked from the prompt owner once it has prefilled,
    // so the prompt's KV is computed and stored once and shared (prefix sharing / prefill-once).
    private readonly Dictionary<string, List<Sequence>> _deferredSiblings = new();

    private long _totalPreemptions;
    private long _totalFinishedRequests;
    private bool _disposed;

    /// <summary>Creates a continuous-batching engine over the given model runner.</summary>
    public ContinuousBatchingEngine(IServingModelRunner<T> runner, EngineOptions? options = null)
    {
        _runner = runner ?? throw new ArgumentNullException(nameof(runner));
        _options = options ?? new EngineOptions();
        _options.Validate();
        _blocks = new BlockManager(_options.NumKvBlocks, _options.BlockSize);
    }

    /// <inheritdoc/>
    public void AddRequest(GenerationRequest request)
    {
        if (request is null) throw new ArgumentNullException(nameof(request));
        lock (_intakeLock) _pendingAdds.Enqueue(request);
    }

    /// <inheritdoc/>
    public bool AbortRequest(string requestId)
    {
        if (string.IsNullOrEmpty(requestId)) return false;
        lock (_intakeLock)
        {
            _pendingAborts.Add(requestId);
            // Report found if it is anywhere we know about (pending, waiting, or running).
            return _byRequest.ContainsKey(requestId)
                || _pendingAdds.Any(r => r.RequestId == requestId);
        }
    }

    /// <inheritdoc/>
    public bool HasUnfinishedRequests
    {
        get
        {
            lock (_intakeLock)
                return _pendingAdds.Count > 0 || _waiting.Count > 0 || _running.Count > 0;
        }
    }

    /// <inheritdoc/>
    public IReadOnlyList<RequestOutput> Step()
    {
        var touchedRequests = new HashSet<string>();
        DrainIntake(touchedRequests);

        // Schedule this step's batch: running sequences decode (priority), then admit waiting prompts. Each
        // scheduled sequence reserves the KV slot for the token it will generate at schedule time (before
        // prefill admission consumes free blocks), so the post-sample append can never run out of memory.
        var scheduled = new List<Sequence>();
        int batchTokens = 0;

        ScheduleDecode(scheduled, ref batchTokens);
        SchedulePrefill(scheduled, ref batchTokens, touchedRequests);

        if (scheduled.Count == 0)
            return touchedRequests.Count == 0 ? Array.Empty<RequestOutput>() : CollectOutputs(touchedRequests);

        // Build executions and run the model once for the whole batch.
        var executions = new List<SequenceExecution<T>>(scheduled.Count);
        foreach (var seq in scheduled)
        {
            bool isPrefill = seq.NumComputedTokens == 0;
            executions.Add(new SequenceExecution<T>(
                seq.SequenceId,
                seq.TokenIds,
                seq.NumComputedTokens,
                _blocks.GetBlockTable(seq.SequenceId),
                Array.Empty<BlockCopy>(),
                isPrefill));
        }

        var logits = _runner.Execute(executions);
        if (logits is null || logits.Count != scheduled.Count)
            throw new InvalidOperationException(
                $"Model runner returned {logits?.Count ?? 0} logit vectors for a batch of {scheduled.Count}.");

        // Sample + advance each scheduled sequence.
        for (int i = 0; i < scheduled.Count; i++)
        {
            AdvanceSequence(scheduled[i], logits[i]);
            touchedRequests.Add(scheduled[i].Request.RequestId);
        }

        return CollectOutputs(touchedRequests);
    }

    /// <inheritdoc/>
    public EngineStatistics GetStatistics()
    {
        lock (_intakeLock)
        {
            return new EngineStatistics
            {
                RunningSequences = _running.Count,
                WaitingSequences = _waiting.Count,
                SwappedSequences = 0, // recompute preemption returns sequences to Waiting, counted there
                KvCacheUsage = _blocks.Usage,
                TotalKvBlocks = _blocks.TotalBlocks,
                FreeKvBlocks = _blocks.NumFreeBlocks,
                TotalPreemptions = _totalPreemptions,
                TotalFinishedRequests = _totalFinishedRequests,
            };
        }
    }

    // ---- Intake ---------------------------------------------------------------------------

    private void DrainIntake(HashSet<string> touchedRequests)
    {
        lock (_intakeLock)
        {
            while (_pendingAdds.Count > 0)
                AdmitNewRequest(_pendingAdds.Dequeue());

            if (_pendingAborts.Count > 0)
            {
                foreach (string requestId in _pendingAborts)
                    if (ApplyAbort(requestId)) touchedRequests.Add(requestId);
                _pendingAborts.Clear();
            }
        }
    }

    private void AdmitNewRequest(GenerationRequest request)
    {
        int n = request.SamplingParameters.N;
        var seqs = new List<Sequence>(n);
        for (int i = 0; i < n; i++)
        {
            var seq = new Sequence(request, i);
            seqs.Add(seq);
            _rngBySequence[seq.SequenceId] = request.SamplingParameters.Seed is { } s
                ? RandomHelper.CreateSeededRandom(s + i) // decorrelate parallel samples
                : RandomHelper.CreateSecureRandom();
        }
        _byRequest[request.RequestId] = seqs;

        // Only the prompt owner (index 0) is queued now. Its siblings are deferred and forked from its prompt
        // KV once it has prefilled, so the shared prompt is processed and stored exactly once.
        _waiting.AddLast(seqs[0]);
        if (n > 1) _deferredSiblings[request.RequestId] = seqs.GetRange(1, n - 1);
    }

    private bool ApplyAbort(string requestId)
    {
        if (!_byRequest.TryGetValue(requestId, out var seqs)) return false;
        _deferredSiblings.Remove(requestId); // drop any not-yet-forked siblings
        foreach (var seq in seqs)
        {
            if (seq.State.IsFinished()) continue;
            RemoveFromQueues(seq);
            _blocks.Free(seq.SequenceId);
            _rngBySequence.Remove(seq.SequenceId);
            seq.Finish(SequenceState.FinishedAborted, "abort");
        }
        return true;
    }

    private void RemoveFromQueues(Sequence seq)
    {
        _running.Remove(seq);
        var node = _waiting.Find(seq);
        if (node is not null) _waiting.Remove(node);
    }

    // ---- Scheduling -----------------------------------------------------------------------

    private void ScheduleDecode(List<Sequence> scheduled, ref int batchTokens)
    {
        // Oldest-first priority. Snapshot because preemption mutates _running.
        var scheduledSet = new HashSet<Sequence>();
        var snapshot = new List<Sequence>(_running);
        foreach (var seq in snapshot)
        {
            if (seq.State != SequenceState.Running) continue; // may have been preempted as a victim

            // Reserve the KV slot for the token this seq will generate this step; preempt other sequences
            // (recompute) if the pool is full.
            bool canRun = true;
            while (!_blocks.CanAppend(seq.SequenceId, 1))
            {
                var victim = FindPreemptionVictim(scheduledSet, seq);
                if (victim is null) { canRun = false; break; } // nothing left to free
                PreemptRecompute(victim);
                if (ReferenceEquals(victim, seq)) { canRun = false; break; } // preempted ourselves
            }
            if (!canRun || seq.State != SequenceState.Running) continue;

            if (batchTokens + 1 > _options.MaxBatchedTokens && scheduled.Count > 0) continue; // budget; next step

            _blocks.Append(seq.SequenceId, 1); // reserve the generation slot now
            scheduled.Add(seq);
            scheduledSet.Add(seq);
            batchTokens += 1;
        }
    }

    // Newest un-scheduled running sequence (may be the caller itself, meaning "preempt yourself").
    private Sequence? FindPreemptionVictim(HashSet<Sequence> scheduledSet, Sequence current)
    {
        for (int i = _running.Count - 1; i >= 0; i--)
        {
            var cand = _running[i];
            if (!scheduledSet.Contains(cand)) return cand;
        }
        return null;
    }

    private void SchedulePrefill(List<Sequence> scheduled, ref int batchTokens, HashSet<string> touchedRequests)
    {
        while (_waiting.First is { } node)
        {
            var seq = node.Value;
            int needed = seq.Length; // fresh prompt, or prompt+generated for a recompute-preempted sequence

            // A sequence whose prompt (plus one generation slot) can never fit the whole pool is a hard
            // failure — finish it aborted so the loop does not stall forever.
            if (_blocks.BlocksForTokens(needed + 1) > _blocks.TotalBlocks)
            {
                _waiting.Remove(node);
                _rngBySequence.Remove(seq.SequenceId);
                seq.Finish(SequenceState.FinishedAborted, "prompt_too_long");
                touchedRequests.Add(seq.Request.RequestId);
                continue;
            }

            if (_running.Count >= _options.MaxNumSequences) break;
            // Need room for the prompt AND the first generated token's slot.
            if (_blocks.BlocksForTokens(needed + 1) > _blocks.NumFreeBlocks) break; // wait for memory
            if (batchTokens + needed > _options.MaxBatchedTokens && scheduled.Count > 0) break; // budget

            _waiting.Remove(node);
            _blocks.Allocate(seq.SequenceId, needed);
            _blocks.Append(seq.SequenceId, 1); // reserve the generation slot now
            seq.State = SequenceState.Running;
            seq.NumComputedTokens = 0; // prefill (also re-prefill for recomputed sequences)
            _running.Add(seq);
            scheduled.Add(seq);
            batchTokens += needed;
        }
    }

    private void PreemptRecompute(Sequence victim)
    {
        _blocks.Free(victim.SequenceId);
        victim.NumComputedTokens = 0;
        victim.State = SequenceState.Waiting;
        _running.Remove(victim);
        _waiting.AddFirst(victim); // resume as soon as memory allows
        _totalPreemptions++;
    }

    // ---- Advancing a sequence -------------------------------------------------------------

    private void AdvanceSequence(Sequence seq, Vector<T> logits)
    {
        int lengthBefore = seq.Length;
        bool wasPrefill = seq.NumComputedTokens == 0;
        seq.NumComputedTokens = lengthBefore; // all current tokens now have cached KV

        // The prompt owner just finished prefill: fork its now-computed prompt KV to any deferred siblings,
        // which sample their first token from these same final-position logits (prompt computed once).
        if (wasPrefill && seq.SequenceIndex == 0)
            SpawnDeferredSiblings(seq, logits);

        int tokenId = LogitsSampler.Sample(logits, seq.Request.SamplingParameters, seq.TokenIds, _rngBySequence[seq.SequenceId]);
        // The KV slot for this token was already reserved at schedule time; appending the token now keeps the
        // block-manager length (seq.Length + 1 reserved) in step with the sequence.
        seq.AppendToken(tokenId);

        var (finished, terminalState, reason) = EvaluateStop(seq, tokenId);
        if (finished)
        {
            _blocks.Free(seq.SequenceId);
            _running.Remove(seq);
            _rngBySequence.Remove(seq.SequenceId);
            seq.Finish(terminalState, reason);
        }
    }

    private void SpawnDeferredSiblings(Sequence owner, Vector<T> promptFinalLogits)
    {
        if (!_deferredSiblings.TryGetValue(owner.Request.RequestId, out var siblings)) return;
        _deferredSiblings.Remove(owner.Request.RequestId);

        int promptLen = owner.PromptLength;
        // Clean sharing needs the prompt to fill whole blocks, so the shared blocks hold exactly the prompt and
        // the first generated token lands in a fresh block (no copy-on-write). Otherwise fall back to giving
        // each sibling its own prompt allocation (independent prefill via the waiting queue).
        bool blockAligned = promptLen % _options.BlockSize == 0;

        foreach (var sibling in siblings)
        {
            // The shared prompt costs no new blocks; the sibling's first generated token needs one slot.
            if (blockAligned && _blocks.NumFreeBlocks >= 1)
            {
                _blocks.ForkPrefix(owner.SequenceId, sibling.SequenceId, promptLen); // shares prompt KV

                // Sample the sibling's first token from the SAME prompt-final logits (prompt computed once);
                // greedy ⇒ same as the owner, stochastic ⇒ independent via the sibling's own RNG.
                int firstToken = LogitsSampler.Sample(
                    promptFinalLogits, sibling.Request.SamplingParameters, sibling.TokenIds, _rngBySequence[sibling.SequenceId]);
                sibling.AppendToken(firstToken);

                var (finished, terminalState, reason) = EvaluateStop(sibling, firstToken);
                if (finished)
                {
                    _blocks.Free(sibling.SequenceId);
                    _rngBySequence.Remove(sibling.SequenceId);
                    sibling.Finish(terminalState, reason);
                    continue;
                }

                _blocks.Append(sibling.SequenceId, 1);  // reserve the first generated token's KV slot
                sibling.NumComputedTokens = promptLen;  // prompt KV cached (shared); the new token computes next step
                sibling.State = SequenceState.Running;
                _running.Add(sibling);
            }
            else
            {
                sibling.State = SequenceState.Waiting;
                sibling.NumComputedTokens = 0;
                _waiting.AddLast(sibling); // independent prefill
            }
        }
    }

    private (bool finished, SequenceState state, string reason) EvaluateStop(Sequence seq, int tokenId)
    {
        var p = seq.Request.SamplingParameters;
        int generated = seq.GeneratedLength;

        bool minReached = generated >= p.MinTokens;
        if (minReached)
        {
            if (!p.IgnoreEos && _options.EosTokenId is { } eos && tokenId == eos)
                return (true, SequenceState.FinishedStopped, "stop");
            if (p.StopTokenIds is { } stops && stops.Contains(tokenId))
                return (true, SequenceState.FinishedStopped, "stop");
        }
        if (generated >= p.MaxTokens)
            return (true, SequenceState.FinishedLengthCapped, "length");

        return (false, SequenceState.Running, string.Empty);
    }

    // ---- Output assembly ------------------------------------------------------------------

    private IReadOnlyList<RequestOutput> CollectOutputs(HashSet<string> touchedRequests)
    {
        var outputs = new List<RequestOutput>(touchedRequests.Count);
        foreach (string requestId in touchedRequests)
        {
            if (!_byRequest.TryGetValue(requestId, out var seqs)) continue;

            var completions = new List<CompletionOutput>(seqs.Count);
            bool allFinished = true;
            foreach (var seq in seqs)
            {
                int promptLen = seq.PromptLength;
                var generated = seq.Length > promptLen
                    ? seq.TokenIds.Skip(promptLen).ToArray()
                    : Array.Empty<int>();
                bool seqFinished = seq.State.IsFinished();
                allFinished &= seqFinished;
                completions.Add(new CompletionOutput(seq.SequenceIndex, generated, seqFinished, seq.FinishReason));
            }

            outputs.Add(new RequestOutput(requestId, seqs[0].Request.PromptTokenIds, completions, allFinished));

            if (allFinished)
            {
                _byRequest.Remove(requestId);
                _totalFinishedRequests++;
            }
        }
        return outputs;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_runner is IDisposable d) d.Dispose();
    }
}
