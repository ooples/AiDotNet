using System.Diagnostics;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Evolution;

public sealed partial class EvolutionEngine<TGenome>
{
    private long _completionSequence;

    private async Task<PreparedProposal> PrepareSeedAsync(TGenome genome, CancellationToken cancellationToken)
    {
        long evaluationId = AllocateProposalId();
        int island = (int)(evaluationId % _islands.Length);
        var lineage = new EvolutionLineage(null, null, "seed", _refiner?.Id, 0, island, (ulong)evaluationId);
        return new PreparedProposal(await PrepareGenomeAsync(genome, evaluationId, island, lineage, cancellationToken).ConfigureAwait(false));
    }

    private async Task<PreparedProposal?> PrepareVariationAsync(CancellationToken cancellationToken)
    {
        long evaluationId = _nextEvaluationId;
        int island = (int)(evaluationId % _islands.Length);
        IEvolutionArchive<TGenome>? sourceArchive = FindSelectionArchive(island);
        if (sourceArchive is null) return null;

        StableRandom proposalRandom = StableRandom.CreateStream(_options.Seed, unchecked((ulong)evaluationId * 8UL));
        EvolutionSelection<TGenome>? selection = _selection.Select(sourceArchive, proposalRandom, _options.InspirationCount);
        if (selection is null) return null;
        long allocatedId = AllocateProposalId();
        if (allocatedId != evaluationId) throw new InvalidOperationException("Proposal identity allocation was not sequential.");
        long generation = ++_generation;
        string[] inspirationIds = selection.Inspirations.Select(entry => entry.Evaluation.GenomeId).ToArray();
        var lineage = new EvolutionLineage(
            new[] { selection.Parent.Evaluation.GenomeId },
            inspirationIds,
            _variation.Id,
            _refiner?.Id,
            generation,
            island,
            (ulong)evaluationId);

        TGenome proposed;
        try
        {
            var context = new EvolutionVariationContext<TGenome>(selection.Parent, selection.Inspirations,
                proposalRandom, generation, island);
            proposed = await _variation.ProposeAsync(context, cancellationToken).ConfigureAwait(false);
            if (proposed is null) throw new InvalidOperationException("The variation operator returned null.");
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception)
        {
            return new PreparedProposal(CreatePreEvaluationFailure(evaluationId, island, lineage, "variation_failure"));
        }

        return new PreparedProposal(await PrepareGenomeAsync(proposed, evaluationId, island, lineage, cancellationToken).ConfigureAwait(false));
    }

    private async Task<WorkItem> PrepareGenomeAsync(TGenome genome, long evaluationId, int island,
        EvolutionLineage lineage, CancellationToken cancellationToken)
    {
        try
        {
            if (_refiner is not null)
            {
                StableRandom refinerRandom = StableRandom.CreateStream(_options.Seed, unchecked((ulong)evaluationId * 8UL + 1UL));
                genome = await _refiner.RefineAsync(genome,
                    new EvolutionRefinementContext(evaluationId, refinerRandom), cancellationToken).ConfigureAwait(false);
                if (genome is null) throw new InvalidOperationException("The candidate refiner returned null.");
            }

            EvolutionCanonicalGenome<TGenome> canonical = await _task.CanonicalizeAsync(genome, cancellationToken).ConfigureAwait(false);
            if (canonical is null) throw new InvalidOperationException("The evolution task returned a null canonical genome.");
            var candidate = new EvolutionCandidate<TGenome>(evaluationId, canonical, lineage);
            await NotifyAsync(new EvolutionEvent<TGenome>(EvolutionEventKind.Proposed, NextEventSequence(), candidate),
                cancellationToken).ConfigureAwait(false);

            if (_options.EnableEvaluationCache && _cache.TryGetValue(canonical.Id, out EvolutionTaskResult? cached))
            {
                return new WorkItem(lineage)
                {
                    EvaluationId = evaluationId,
                    Island = island,
                    Candidate = candidate,
                    Result = CopyWithZeroCost(cached),
                    CacheStatus = EvolutionCacheStatus.Hit,
                    CompletionOrder = Interlocked.Increment(ref _completionSequence)
                };
            }

            if (!_seen.Add(canonical.Id))
            {
                return new WorkItem(lineage)
                {
                    EvaluationId = evaluationId,
                    Island = island,
                    Candidate = candidate,
                    Result = new EvolutionTaskResult(EvolutionEvaluationStatus.Duplicate),
                    CacheStatus = EvolutionCacheStatus.NotChecked,
                    CompletionOrder = Interlocked.Increment(ref _completionSequence)
                };
            }

            return new WorkItem(lineage)
            {
                EvaluationId = evaluationId,
                Island = island,
                Candidate = candidate,
                CacheStatus = _options.EnableEvaluationCache ? EvolutionCacheStatus.Miss : EvolutionCacheStatus.NotChecked,
                RequiresEvaluation = true,
                AddedToSeen = true
            };
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception)
        {
            return CreatePreEvaluationFailure(evaluationId, island, lineage, "canonicalization_failure");
        }
    }

    private WorkItem CreatePreEvaluationFailure(long evaluationId, int island, EvolutionLineage lineage, string code)
    {
        return new WorkItem(lineage)
        {
            EvaluationId = evaluationId,
            Island = island,
            Result = EvolutionTaskResult.Failed(code, "Candidate preparation failed."),
            CacheStatus = EvolutionCacheStatus.NotChecked,
            CompletionOrder = Interlocked.Increment(ref _completionSequence)
        };
    }

    private long AllocateProposalId()
    {
        long id = _nextEvaluationId++;
        _proposals++;
        return id;
    }

    private IEvolutionArchive<TGenome>? FindSelectionArchive(int preferredIsland)
    {
        if (_islands[preferredIsland].Count > 0) return _islands[preferredIsland];
        for (int offset = 1; offset < _islands.Length; offset++)
        {
            IEvolutionArchive<TGenome> archive = _islands[(preferredIsland + offset) % _islands.Length];
            if (archive.Count > 0) return archive;
        }
        return null;
    }

    private async Task EvaluateBatchAsync(List<WorkItem> batch, CancellationToken cancellationToken)
    {
        List<WorkItem> pending = batch.Where(item => item.RequiresEvaluation).OrderBy(item => item.EvaluationId).ToList();
        while (pending.Count > 0 && _evaluationAttempts < _options.MaxEvaluationAttempts)
        {
            int available = (int)Math.Min(pending.Count, _options.MaxEvaluationAttempts - _evaluationAttempts);
            WorkItem[] round = pending.Take(available).ToArray();
            pending.RemoveRange(0, available);
            foreach (WorkItem item in round)
            {
                item.AttemptCount++;
                _evaluationAttempts++;
            }

            using (var semaphore = new SemaphoreSlim(_options.MaxDegreeOfParallelism, _options.MaxDegreeOfParallelism))
            {
                Task[] tasks = round.Select(item => EvaluateWithSlotAsync(item, semaphore, cancellationToken)).ToArray();
                await Task.WhenAll(tasks).ConfigureAwait(false);
            }

            foreach (WorkItem item in round.OrderBy(item => item.EvaluationId))
            {
                if (IsRetryable(item.Result) && item.AttemptCount <= _options.MaxRetries &&
                    _evaluationAttempts + pending.Count < _options.MaxEvaluationAttempts)
                {
                    pending.Add(item);
                }
                else
                {
                    item.RequiresEvaluation = false;
                }
            }
            pending = pending.OrderBy(item => item.EvaluationId).ToList();
        }

        foreach (WorkItem unevaluated in pending)
        {
            unevaluated.RequiresEvaluation = false;
            unevaluated.Result = new EvolutionTaskResult(EvolutionEvaluationStatus.Skipped,
                diagnostics: new[] { new EvolutionDiagnostic("budget_exhausted", "Evaluator budget was exhausted before dispatch.") });
            unevaluated.CompletionOrder = Interlocked.Increment(ref _completionSequence);
        }
    }

    private async Task EvaluateWithSlotAsync(WorkItem item, SemaphoreSlim semaphore, CancellationToken cancellationToken)
    {
        await semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            Stopwatch timer = Stopwatch.StartNew();
            EvolutionTaskResult result = await EvaluateAttemptAsync(item, cancellationToken).ConfigureAwait(false);
            timer.Stop();
            item.Elapsed += timer.Elapsed;
            AccumulateAttemptMetadata(item, result);
            item.Result = result;
            item.CompletionOrder = Interlocked.Increment(ref _completionSequence);
        }
        finally
        {
            semaphore.Release();
        }
    }

    private async Task<EvolutionTaskResult> EvaluateAttemptAsync(WorkItem item, CancellationToken cancellationToken)
    {
        if (item.Candidate is null) return EvolutionTaskResult.Failed("missing_candidate", "The evaluator candidate was unavailable.");
        using (var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
        {
            if (_options.EvaluationTimeout.HasValue) linked.CancelAfter(_options.EvaluationTimeout.Value);
            try
            {
                var context = new EvolutionEvaluationContext(item.EvaluationId, _options.Seed,
                    unchecked((ulong)item.EvaluationId * 8UL + 2UL), item.AttemptCount);
                EvolutionTaskResult result = await _task.EvaluateAsync(item.Candidate, context, linked.Token).ConfigureAwait(false);
                return result ?? EvolutionTaskResult.Failed("null_result", "The task returned a null evaluation result.");
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                return new EvolutionTaskResult(EvolutionEvaluationStatus.Canceled);
            }
            catch (OperationCanceledException) when (linked.IsCancellationRequested)
            {
                return new EvolutionTaskResult(EvolutionEvaluationStatus.TimedOut,
                    diagnostics: new[] { new EvolutionDiagnostic("evaluation_timeout", "The evaluator exceeded its cooperative timeout.") });
            }
            catch (Exception exception)
            {
                return new EvolutionTaskResult(EvolutionEvaluationStatus.Failed,
                    diagnostics: new[] { new EvolutionDiagnostic("evaluator_exception",
                        $"Evaluator threw {exception.GetType().Name}.", isRedacted: true) });
            }
        }
    }

    private async Task<bool> CommitBatchAsync(List<WorkItem> batch, CancellationToken cancellationToken)
    {
        IEnumerable<WorkItem> ordered = _options.ExecutionMode == EvolutionExecutionMode.Deterministic
            ? batch.OrderBy(item => item.EvaluationId)
            : batch.OrderBy(item => item.CompletionOrder).ThenBy(item => item.EvaluationId);
        bool failedFast = false;
        foreach (WorkItem item in ordered)
        {
            EvolutionTaskResult result = item.Result ?? EvolutionTaskResult.Failed("missing_result", "No terminal evaluator result was produced.");
            EvolutionEvaluation evaluation = BuildEvaluation(item, result);
            IncrementStatus(evaluation.Status);
            EvolutionArchiveInsertionResult? insertion = null;
            if (evaluation.Status == EvolutionEvaluationStatus.Completed)
            {
                _completedEvaluations++;
                if (item.Candidate is not null)
                {
                    insertion = _islands[item.Island].TryAdd(item.Candidate, evaluation);
                    if (_options.EnableEvaluationCache && item.CacheStatus != EvolutionCacheStatus.Hit)
                        _cache[item.Candidate.CanonicalGenome.Id] = result;
                }
            }
            else if (item.Candidate is not null && !_options.DeduplicateFailedCandidates &&
                     IsFailureLike(evaluation.Status))
            {
                _seen.Remove(item.Candidate.CanonicalGenome.Id);
            }

            if (_selection is IOutcomeAwareEvolutionSelectionPolicy<TGenome> adaptiveSelection)
                adaptiveSelection.Observe(evaluation, insertion);

            if (IsFailureLike(evaluation.Status))
            {
                foreach (EvolutionDiagnostic diagnostic in evaluation.Diagnostics) RetainFailure(diagnostic);
                failedFast |= evaluation.Status != EvolutionEvaluationStatus.Canceled &&
                              _options.FailurePolicy == EvolutionFailurePolicy.FailFast;
            }

            await NotifyAsync(new EvolutionEvent<TGenome>(EvolutionEventKind.Evaluated, NextEventSequence(),
                item.Candidate, evaluation, insertion), cancellationToken).ConfigureAwait(false);
            if (insertion == EvolutionArchiveInsertionResult.Inserted ||
                insertion == EvolutionArchiveInsertionResult.Replaced ||
                insertion == EvolutionArchiveInsertionResult.InsertedWithEviction)
            {
                await NotifyAsync(new EvolutionEvent<TGenome>(EvolutionEventKind.ArchiveChanged, NextEventSequence(),
                    item.Candidate, evaluation, insertion), cancellationToken).ConfigureAwait(false);
            }

            if (_checkpointStore is not null && _options.CheckpointInterval > 0) _commitsSinceCheckpoint++;
        }
        return failedFast;
    }

    private EvolutionEvaluation BuildEvaluation(WorkItem item, EvolutionTaskResult result)
    {
        string genomeId = item.Candidate?.CanonicalGenome.Id ?? $"unavailable:{item.EvaluationId}";
        double costUnits = item.CacheStatus == EvolutionCacheStatus.Hit ? 0 : item.AccumulatedCostUnits;
        IReadOnlyList<EvolutionDiagnostic> diagnostics = item.AttemptCount == 0
            ? result.Diagnostics
            : item.AttemptDiagnostics;
        return new EvolutionEvaluation(
            item.EvaluationId,
            genomeId,
            result.Status,
            result.Quality,
            result.Direction,
            result.Descriptors,
            result.Objectives,
            result.ConstraintViolations,
            new EvolutionEvaluationCost(item.Elapsed, item.AttemptCount, costUnits),
            item.Lineage,
            item.CacheStatus,
            diagnostics,
            _task.VersionHash,
            _task.EvaluatorVersionHash,
            _configurationHash);
    }

    private async Task MigrateIfDueAsync(CancellationToken cancellationToken)
    {
        if (_islands.Length < 2 || _options.MigrationInterval == 0 || _batchesSinceMigration < _options.MigrationInterval)
            return;
        _batchesSinceMigration = 0;
        StableRandom random = StableRandom.CreateStream(_options.Seed, unchecked(0x8000000000000000UL + (ulong)_generation));
        IReadOnlyList<EvolutionMigration<TGenome>> migrations = _migration.CreateMigrations(
            Array.AsReadOnly(_islands), _options.MigrantsPerIsland, random)
            ?? throw new InvalidOperationException("The migration policy returned null.");
        if (migrations.Any(item => item is null))
            throw new InvalidOperationException("The migration policy returned a null transfer.");
        foreach (IGrouping<int, EvolutionMigration<TGenome>> sourceGroup in migrations.GroupBy(item => item.SourceIsland))
            if (sourceGroup.Count() > _options.MigrantsPerIsland)
                throw new InvalidOperationException("The migration policy exceeded the per-island transfer bound.");
        EvolutionMigration<TGenome>[] orderedMigrations = migrations
            .OrderBy(item => item.SourceIsland)
            .ThenBy(item => item.DestinationIsland)
            .ThenBy(item => item.Entry.Evaluation.GenomeId, StringComparer.Ordinal)
            .ToArray();
        foreach (EvolutionMigration<TGenome> migration in orderedMigrations)
        {
            if (migration.SourceIsland < 0 || migration.DestinationIsland < 0 ||
                migration.SourceIsland >= _islands.Length || migration.DestinationIsland >= _islands.Length ||
                migration.SourceIsland == migration.DestinationIsland)
                throw new InvalidOperationException("The migration policy produced an invalid island index.");
            bool belongsToSource = _islands[migration.SourceIsland].Entries.Any(entry =>
                entry.Cell.Equals(migration.Entry.Cell) &&
                entry.Evaluation.EvaluationId == migration.Entry.Evaluation.EvaluationId &&
                string.Equals(entry.Evaluation.GenomeId, migration.Entry.Evaluation.GenomeId, StringComparison.Ordinal));
            if (!belongsToSource)
                throw new InvalidOperationException("The migration policy produced an entry that is not in its source island.");
        }
        foreach (EvolutionMigration<TGenome> migration in orderedMigrations)
        {
            _islands[migration.DestinationIsland].TryAdd(migration.Entry.Candidate, migration.Entry.Evaluation);
        }
        await NotifyAsync(new EvolutionEvent<TGenome>(EvolutionEventKind.Migrated, NextEventSequence(),
            message: $"{migrations.Count} elite transfers"), cancellationToken).ConfigureAwait(false);
    }

    private void IncrementStatus(EvolutionEvaluationStatus status)
    {
        _statusCounts.TryGetValue(status, out long current);
        _statusCounts[status] = current + 1;
    }

    private static bool IsRetryable(EvolutionTaskResult? result) => result is not null && IsFailureLike(result.Status);

    private static bool IsFailureLike(EvolutionEvaluationStatus status) =>
        status == EvolutionEvaluationStatus.Failed ||
        status == EvolutionEvaluationStatus.TimedOut ||
        status == EvolutionEvaluationStatus.Canceled;

    private static void AccumulateAttemptMetadata(WorkItem item, EvolutionTaskResult result)
    {
        if (result.CostUnits > double.MaxValue - item.AccumulatedCostUnits)
        {
            item.AccumulatedCostUnits = double.MaxValue;
            AddAttemptDiagnostic(item, new EvolutionDiagnostic("cost_units_saturated",
                "Accumulated evaluator cost exceeded the finite numeric range."));
        }
        else
        {
            item.AccumulatedCostUnits += result.CostUnits;
        }

        foreach (EvolutionDiagnostic diagnostic in result.Diagnostics) AddAttemptDiagnostic(item, diagnostic);
    }

    private static void AddAttemptDiagnostic(WorkItem item, EvolutionDiagnostic diagnostic)
    {
        const int maximumDiagnostics = 64;
        if (item.AttemptDiagnostics.Count < maximumDiagnostics)
        {
            item.AttemptDiagnostics.Add(diagnostic);
            return;
        }

        if (item.AttemptDiagnostics[maximumDiagnostics - 1].Code != "diagnostics_truncated")
        {
            item.AttemptDiagnostics[maximumDiagnostics - 1] = new EvolutionDiagnostic(
                "diagnostics_truncated", "Additional retry diagnostics were omitted to preserve the public bound.");
        }
    }

    private static EvolutionTaskResult CopyWithZeroCost(EvolutionTaskResult result) => new(
        result.Status, result.Quality, result.Direction, result.Descriptors, result.Objectives,
        result.ConstraintViolations, 0, result.Diagnostics);
}
