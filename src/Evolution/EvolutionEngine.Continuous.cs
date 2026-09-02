using System.Diagnostics;
using AiDotNet.Enums;

namespace AiDotNet.Evolution;

public sealed partial class EvolutionEngine<TGenome>
{
    /// <summary>
    /// Runs the search as a continuously refilled window of evaluations instead of as a sequence of batches.
    /// </summary>
    /// <param name="seeds">The caller's seed genomes.</param>
    /// <param name="seedIndex">How many seeds a resumed run already consumed.</param>
    /// <param name="runTimer">The stopwatch the time limit is measured on.</param>
    /// <param name="cancellationToken">Cancels the run.</param>
    /// <returns>Why the run stopped.</returns>
    /// <remarks>
    /// <para>
    /// The window holds at most <c>MaxInFlight</c> evaluations, defaulting to the worker count. Each time the oldest
    /// evaluation in the window finishes it is committed on its own and exactly one replacement proposal is prepared,
    /// so no worker waits for the rest of a batch and no proposal is more than one window behind the archive. Pairing
    /// each commit with exactly one preparation is also what keeps the mode deterministic: the proposal for evaluation
    /// N is prepared once evaluation N minus the window size has committed, whatever the worker count or the
    /// evaluator's timing.
    /// </para>
    /// <para>
    /// Three rules keep that schedule honest. Commits follow evaluation-id order under
    /// <see cref="EvolutionExecutionMode.Deterministic"/>, so a finished evaluation waits for its predecessors. The
    /// oldest evaluation is dispatched even when its island is at its in-flight quota, so a quota can delay work but
    /// never deadlock it. And a checkpoint is written only once the window has drained, because the run counters a
    /// checkpoint records already count every in-flight proposal, so writing one mid-window would claim proposals
    /// whose outcome was never committed.
    /// </para>
    /// </remarks>
    private async Task<EvolutionStopReason> RunContinuousLoopAsync(TGenome[] seeds, int seedIndex, Stopwatch runTimer,
        CancellationToken cancellationToken)
    {
        var state = new ContinuousState(seeds, seedIndex,
            Math.Max(1, _options.MaxInFlight > 0 ? _options.MaxInFlight : _options.MaxDegreeOfParallelism));

        using var semaphore = new SemaphoreSlim(_options.MaxDegreeOfParallelism, _options.MaxDegreeOfParallelism);
        BatchTransaction transaction = CaptureBatchTransaction();
        try
        {
            while (true)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (Volatile.Read(ref _stopRequested) != 0 && state.InFlight.Count == 0) return EvolutionStopReason.Canceled;

                await FillWindowAsync(state, semaphore, runTimer, cancellationToken).ConfigureAwait(false);

                if (state.InFlight.Count == 0)
                {
                    if (state.DrainingForCheckpoint)
                    {
                        state.DrainingForCheckpoint = false;
                        CaptureSafeState(state.Seeds, state.CommittedSeeds);
                        await SaveCheckpointAsync(force: false, cancellationToken).ConfigureAwait(false);
                        transaction = CaptureBatchTransaction();
                        if (state.Stop is null) continue;
                    }
                    return state.Stop ?? StopReasonWithNothingInFlight();
                }

                DispatchWaiting(state, semaphore, cancellationToken);
                if (state.Running.Count > 0)
                {
                    await Task.WhenAny(state.Running.Values).ConfigureAwait(false);
                    await ReapFinishedAsync(state, semaphore, cancellationToken).ConfigureAwait(false);
                }

                while (TryTakeCommittable(state, out WorkItem? ready) && ready is not null)
                {
                    state.InFlight.Remove(ready);
                    if (ready.IsSeed) state.CommittedSeeds++;

                    // A single-item commit reuses the batch commit path, so archives, cache, elites, history,
                    // observers, and the failure policy all behave exactly as they do in batch mode.
                    bool failedFast = await CommitBatchAsync(new List<WorkItem> { ready }, CancellationToken.None)
                        .ConfigureAwait(false);
                    UpdateEarlyStopping(1);
                    await MigrateIfBatchBoundaryAsync(ready.EvaluationId).ConfigureAwait(false);
                    transaction = CaptureBatchTransaction();

                    if (failedFast) state.Stop ??= EvolutionStopReason.CandidateFailure;
                    else if (IsTargetReached()) state.Stop ??= EvolutionStopReason.TargetReached;
                    else if (IsEarlyStopped()) state.Stop ??= EvolutionStopReason.EarlyStopped;
                    if (state.Stop is not null) break;

                    if (_checkpointStore is not null && _options.CheckpointInterval > 0 &&
                        _commitsSinceCheckpoint >= _options.CheckpointInterval)
                    {
                        state.DrainingForCheckpoint = true;
                        break;
                    }

                    // Exactly one replacement per commit: that pairing is what makes the schedule reproducible.
                    await FillOneAsync(state, semaphore, runTimer, cancellationToken).ConfigureAwait(false);
                }
            }
        }
        catch (OperationCanceledException)
        {
            RestoreBatchTransaction(transaction, state.InFlight);
            throw;
        }
    }

    /// <summary>Reports why a continuous run has nothing left to do.</summary>
    private EvolutionStopReason StopReasonWithNothingInFlight()
    {
        if (_evaluationAttempts >= _options.MaxEvaluationAttempts) return EvolutionStopReason.EvaluationBudgetReached;
        if (_proposals >= _options.MaxProposals) return EvolutionStopReason.ProposalBudgetReached;
        if (_generation >= _options.MaxGenerations) return EvolutionStopReason.GenerationLimitReached;
        return EvolutionStopReason.NoCandidates;
    }

    /// <summary>Fills the window up to its size, stopping at the first proposal that cannot be made.</summary>
    private async Task FillWindowAsync(ContinuousState state, SemaphoreSlim semaphore, Stopwatch runTimer,
        CancellationToken cancellationToken)
    {
        while (state.InFlight.Count < state.Window && state.Stop is null && !state.DrainingForCheckpoint &&
               await FillOneAsync(state, semaphore, runTimer, cancellationToken).ConfigureAwait(false))
        {
        }
    }

    /// <summary>Prepares and admits exactly one proposal into the window.</summary>
    /// <returns><c>true</c> when a proposal was admitted; <c>false</c> when nothing more can be proposed now.</returns>
    private async Task<bool> FillOneAsync(ContinuousState state, SemaphoreSlim semaphore, Stopwatch runTimer,
        CancellationToken cancellationToken)
    {
        if (state.Stop is not null || state.DrainingForCheckpoint || state.InFlight.Count >= state.Window) return false;
        cancellationToken.ThrowIfCancellationRequested();
        EvolutionStopReason? limit = GetLimitStopReason(runTimer);
        if (limit.HasValue)
        {
            state.Stop = limit.Value;
            return false;
        }

        WorkItem item;
        bool isSeed = state.SeedIndex < state.Seeds.Length;
        if (isSeed)
        {
            PreparedProposal seeded = await PrepareSeedAsync(state.Seeds[state.SeedIndex], cancellationToken)
                .ConfigureAwait(false);
            state.SeedIndex++;
            item = seeded.Item;
        }
        else
        {
            if (_generation >= _options.MaxGenerations)
            {
                state.Stop = EvolutionStopReason.GenerationLimitReached;
                return false;
            }
            PreparedProposal? prepared = await PrepareVariationAsync(cancellationToken).ConfigureAwait(false);
            if (prepared is null) return false;
            item = prepared.Item;
        }

        item.IsSeed = isSeed;
        state.InFlight.Add(item);
        DispatchWaiting(state, semaphore, cancellationToken);
        return true;
    }

    /// <summary>Starts every admitted evaluation the worker pool and the island quotas currently allow.</summary>
    /// <remarks>
    /// The oldest admitted evaluation is exempt from the island quota. Commits follow identifier order, so holding the
    /// oldest one back would stop every commit, which would in turn stop the quota from ever freeing; exempting it
    /// keeps the quota a throttle rather than a deadlock.
    /// </remarks>
    private void DispatchWaiting(ContinuousState state, SemaphoreSlim semaphore, CancellationToken cancellationToken)
    {
        int quota = _options.MaxInFlightPerIsland;
        List<WorkItem> ordered = state.InFlight.OrderBy(item => item.EvaluationId).ToList();
        var perIsland = new int[_islands.Length];
        foreach (WorkItem item in ordered)
            if (state.Running.ContainsKey(item.EvaluationId)) perIsland[item.Island]++;

        bool first = true;
        foreach (WorkItem item in ordered)
        {
            bool isOldest = first;
            first = false;
            if (!item.RequiresEvaluation || state.Running.ContainsKey(item.EvaluationId)) continue;

            if (_evaluationAttempts >= _options.MaxEvaluationAttempts)
            {
                // Only give up on the budget once nothing is still running, because a cascade stage that screens a
                // candidate out refunds its attempt when it completes and may make room after all.
                if (state.Running.Count > 0) continue;
                item.RequiresEvaluation = false;
                item.Result = new EvolutionTaskResult(EvolutionEvaluationStatus.Skipped,
                    diagnostics: new[] { new EvolutionDiagnostic("budget_exhausted", "Evaluator budget was exhausted before dispatch.") });
                item.CompletionOrder = Interlocked.Increment(ref _completionSequence);
                continue;
            }

            if (!isOldest && quota > 0 && perIsland[item.Island] >= quota) continue;

            item.AttemptCount++;
            _evaluationAttempts++;
            perIsland[item.Island]++;
            state.Running[item.EvaluationId] = EvaluateAfterDelayAsync(item, RetryDelayForAttempt(item.AttemptCount),
                semaphore, cancellationToken);
        }
    }

    /// <summary>Waits out any retry backoff and then evaluates one item inside a worker slot.</summary>
    private async Task EvaluateAfterDelayAsync(WorkItem item, TimeSpan delay, SemaphoreSlim semaphore,
        CancellationToken cancellationToken)
    {
        if (delay > TimeSpan.Zero) await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
        await EvaluateWithSlotAsync(item, semaphore, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Collects finished evaluations, refunds screened-out attempts, and leaves retryable ones queued.</summary>
    private async Task ReapFinishedAsync(ContinuousState state, SemaphoreSlim semaphore,
        CancellationToken cancellationToken)
    {
        long[] finished = state.Running.Where(pair => pair.Value.IsCompleted).Select(pair => pair.Key)
            .OrderBy(id => id).ToArray();
        foreach (long id in finished)
        {
            Task task = state.Running[id];
            state.Running.Remove(id);
            await task.ConfigureAwait(false);

            WorkItem item = state.InFlight.Single(candidate => candidate.EvaluationId == id);
            if (item.CascadeRejectedStage.HasValue && !_options.Cascade.ChargeRejectedStagesToBudget)
                _evaluationAttempts--;

            bool retry = IsRetryable(item.Result) && item.AttemptCount <= _options.MaxRetries &&
                         _evaluationAttempts < _options.MaxEvaluationAttempts;
            if (!retry) item.RequiresEvaluation = false;
        }

        DispatchWaiting(state, semaphore, cancellationToken);
    }

    /// <summary>Picks the next window entry that may be committed, or none when the head is still running.</summary>
    private bool TryTakeCommittable(ContinuousState state, out WorkItem? ready)
    {
        ready = null;
        if (state.InFlight.Count == 0) return false;

        if (_options.ExecutionMode == EvolutionExecutionMode.Deterministic)
        {
            WorkItem head = state.InFlight.OrderBy(item => item.EvaluationId).First();
            if (head.RequiresEvaluation || head.Result is null) return false;
            ready = head;
            return true;
        }

        ready = state.InFlight.Where(item => !item.RequiresEvaluation && item.Result is not null)
            .OrderBy(item => item.CompletionOrder).ThenBy(item => item.EvaluationId).FirstOrDefault();
        return ready is not null;
    }

    /// <summary>Advances the migration counter once per logical batch worth of committed evaluations.</summary>
    /// <param name="committedEvaluationId">The identifier of the evaluation that just committed.</param>
    /// <remarks>
    /// Continuous dispatch has no batches, but <c>MigrationInterval</c> is defined in them, so the boundary is taken
    /// from the committed identifier rather than from a counter. That keeps migration on exactly the same schedule as
    /// batch mode and needs no extra checkpoint state, because identifiers are restored on resume.
    /// </remarks>
    private async Task MigrateIfBatchBoundaryAsync(long committedEvaluationId)
    {
        if (_options.MigrationInterval <= 0 || _islands.Length <= 1) return;
        int batchSize = Math.Max(1, _options.ProposalBatchSize);
        if ((committedEvaluationId + 1) % batchSize != 0) return;
        _batchesSinceMigration++;
        await MigrateIfDueAsync(CancellationToken.None).ConfigureAwait(false);
    }

    /// <summary>The mutable bookkeeping of one continuous run, kept in one place so the loop reads as a pipeline.</summary>
    private sealed class ContinuousState
    {
        public ContinuousState(TGenome[] seeds, int seedIndex, int window)
        {
            Seeds = seeds;
            SeedIndex = seedIndex;
            CommittedSeeds = seedIndex;
            Window = window;
        }

        public TGenome[] Seeds { get; }
        public int Window { get; }
        public int SeedIndex { get; set; }
        public int CommittedSeeds { get; set; }
        public bool DrainingForCheckpoint { get; set; }
        public EvolutionStopReason? Stop { get; set; }
        public List<WorkItem> InFlight { get; } = new();
        public Dictionary<long, Task> Running { get; } = new();
    }
}
