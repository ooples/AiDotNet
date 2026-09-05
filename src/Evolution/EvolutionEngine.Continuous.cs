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
        var state = new ContinuousState(seeds, seedIndex, _options.ResolveInFlightWindow());

        var semaphore = new SemaphoreSlim(_options.MaxDegreeOfParallelism, _options.MaxDegreeOfParallelism);
        try
        {
            while (true)
            {
                cancellationToken.ThrowIfCancellationRequested();

                await FillWindowAsync(state, semaphore, runTimer, cancellationToken).ConfigureAwait(false);

                if (state.InFlight.Count == 0)
                {
                    // The window is drained, which is the only point at which the run's counters describe exactly
                    // what has been committed. Capturing here is what makes the final checkpoint the run's real
                    // final state instead of whatever was last captured, which with checkpointing switched off was
                    // the empty state the run started from.
                    CaptureSafeState(state.Seeds, state.CommittedSeeds);
                    if (state.DrainingForCheckpoint)
                    {
                        state.DrainingForCheckpoint = false;
                        await SaveCheckpointAsync(force: false, cancellationToken).ConfigureAwait(false);
                        if (state.Stop is null && Volatile.Read(ref _stopRequested) == 0) continue;
                    }

                    if (Volatile.Read(ref _stopRequested) != 0) return EvolutionStopReason.Canceled;
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
            RollbackInFlight(state);
            throw;
        }
        finally
        {
            // Evaluations already inside the worker pool release their slot as they unwind, so disposing the
            // semaphore while any of them is still running would fault a task nobody is waiting on. Every run that
            // reaches its time limit takes this path, so it is not an exotic one.
            await DrainRunningAsync(state).ConfigureAwait(false);
            semaphore.Dispose();
        }
    }

    /// <summary>Waits for every dispatched evaluation to unwind, ignoring how each one ended.</summary>
    private static async Task DrainRunningAsync(ContinuousState state)
    {
        if (state.Running.Count == 0) return;
        try
        {
            await Task.WhenAll(state.Running.Values).ConfigureAwait(false);
        }
#pragma warning disable CA1031
        catch (Exception)
#pragma warning restore CA1031
        {
            // Whatever these tasks were doing, the run is already ending and their outcomes are discarded by the
            // rollback. Waiting is only about not disposing the pool from under them.
        }
        finally
        {
            state.Running.Clear();
        }
    }

    /// <summary>Undoes exactly the proposals that were prepared but never committed.</summary>
    /// <remarks>
    /// A snapshot taken after the last commit cannot do this job, because by then the window already holds prepared
    /// proposals whose identifiers and counters are inside the snapshot: restoring it would leave the run counting
    /// proposals whose deduplication entries had just been removed. Undoing each in-flight item by what it actually
    /// consumed is exact, and needs no snapshot at all.
    /// </remarks>
    private void RollbackInFlight(ContinuousState state)
    {
        foreach (WorkItem item in state.InFlight)
        {
            if (item.AddedToSeen && item.Candidate is not null) _seen.Remove(item.Candidate.CanonicalGenome.Id);
            _evaluationAttempts -= item.ChargedAttempts;
            _proposals--;
            _nextEvaluationId--;
            if (item.IsSeed) continue;

            _generation--;
            if (item.Island >= 0 && item.Island < _islandGenerations.Length) _islandGenerations[item.Island]--;
        }

        state.InFlight.Clear();
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

        // A stop request has to reach the proposing side, not only the loop head. Checking it only where the window
        // is already empty means a steady-state run refills forever and never sees the request at all.
        if (Volatile.Read(ref _stopRequested) != 0) return false;

        EvolutionStopReason? limit = GetLimitStopReason(runTimer);
        if (limit.HasValue)
        {
            state.Stop = limit.Value;
            return false;
        }

        // Admitted work that has not been charged yet still consumes the budget, so counting it here keeps admission
        // a function of the window's contents rather than of how far the evaluator happens to have got.
        int uncharged = state.InFlight.Count(item => item.RequiresEvaluation && item.ChargedAttempts == 0);
        if (_evaluationAttempts + uncharged >= _options.MaxEvaluationAttempts)
        {
            state.Stop = EvolutionStopReason.EvaluationBudgetReached;
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
            item.ChargedAttempts++;
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
            {
                _evaluationAttempts--;
                item.ChargedAttempts--;
            }

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
