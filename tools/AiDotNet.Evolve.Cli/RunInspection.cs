using System.Globalization;
using System.Security.Cryptography;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Models.Results;
using Newtonsoft.Json;

namespace AiDotNet.Evolve.Cli;

/// <summary>Copies bounded progress during serialized engine callbacks; IPC never reads a live archive.</summary>
internal sealed class RunInspection(EvolutionRunControl control, CancellationTokenSource cancellation) : IProgramEvolutionArchiveObserver
{
    private readonly object _gate = new();
    private readonly List<IEvolutionArchiveView<ProgramGenome>> _archives = new();
    private readonly HashSet<long> _pending = new();
    private readonly Dictionary<string, long> _statuses = new(StringComparer.Ordinal);
    private readonly Queue<LineageSnapshot> _lineage = new();
    private string _state = "starting";
    private bool _finished, _pendingTruncated, _archivesCopied, _unknownCost;
    private long _events, _terminal, _attempts, _lastEvaluationEvent, _checkpointEvent;
    private long? _checkpointSequence;
    private double _cost;
    private int _archiveCount;
    private int[] _islandCounts = Array.Empty<int>();
    private BestSnapshot? _best;
    private string? _checkpointHash;

    internal sealed record LineageSnapshot(long EvaluationId, string GenomeId, string OperatorId,
        long Generation, int Island, string[] Parents);
    internal sealed record BestSnapshot(string GenomeId, double? Quality, string Direction,
        string TaskVersionHash, string EvaluatorVersionHash, string ConfigurationHash);
    internal sealed record Snapshot(string State, bool Finished, bool StopRequested, long ObservedEvents,
        long SegmentTerminalCandidates, long SegmentEvaluationAttempts, double? SegmentReportedCostUnits,
        bool UnknownConsumption, int? ObservedPendingCandidates, int? BackendQueueDepth,
        string QueueScope, int ArchiveCount, int[] IslandOccupancy, BestSnapshot? BestFeasible,
        string ValidityScope, IReadOnlyDictionary<string, long> SegmentStatuses, LineageSnapshot[] RecentLineage,
        long? CheckpointSequence, string? VerifiedCheckpointSha256, bool Resumable, string ModelUsageScope);

    public void AddArchive(IEvolutionArchiveView<ProgramGenome> archive)
    {
        ArgumentNullException.ThrowIfNull(archive);
        lock (_gate)
        {
            if (_events != 0) throw new InvalidOperationException("Archives must be registered before event delivery.");
            _archives.Add(archive);
        }
    }

    public ValueTask OnEventAsync(EvolutionEvent<ProgramGenome> item, CancellationToken cancellationToken = default)
    {
        lock (_gate)
        {
            _events++;
            if (_state == "starting") _state = "running";
            if (item.Kind == EvolutionEventKind.Proposed && item.Candidate is { } proposed)
            {
                if (_pending.Count < 4096) _pending.Add(proposed.EvaluationId);
                else _pendingTruncated = true;
            }
            if (item.Kind == EvolutionEventKind.Evaluated && item.Evaluation is { } evaluation)
            {
                _lastEvaluationEvent = item.Sequence;
                _pending.Remove(evaluation.EvaluationId);
                _terminal++;
                _attempts += evaluation.Cost.AttemptCount;
                _cost += evaluation.Cost.CostUnits;
                _unknownCost |= !double.IsFinite(_cost);
                string status = evaluation.Status.ToString();
                _statuses[status] = _statuses.TryGetValue(status, out long count) ? count + 1 : 1;
                var lineage = evaluation.Lineage;
                _lineage.Enqueue(new LineageSnapshot(evaluation.EvaluationId, Identity(evaluation.GenomeId),
                    Identity(lineage.VariationOperatorId), lineage.Generation, lineage.Island,
                    lineage.ParentIds.Take(8).Select(Identity).ToArray()));
                if (_lineage.Count > 16) _lineage.Dequeue();
            }
            if (item.Kind == EvolutionEventKind.Checkpointed && item.Message is { } message &&
                message.StartsWith("checkpoint ", StringComparison.Ordinal) &&
                long.TryParse(message.AsSpan(11), NumberStyles.None, CultureInfo.InvariantCulture, out long sequence))
            {
                _checkpointSequence = sequence;
                _checkpointEvent = item.Sequence;
            }
            if (!_archivesCopied || item.Kind is EvolutionEventKind.ArchiveChanged or EvolutionEventKind.Checkpointed or EvolutionEventKind.Stopped)
                CopyArchives();
        }
        return default;
    }

    private void CopyArchives()
    {
        _archivesCopied = true;
        _archiveCount = _archives.Sum(archive => archive.Count);
        _islandCounts = _archives.Take(128).Select(archive => archive.Count).ToArray();
        EvolutionArchiveEntry<ProgramGenome>? best = null;
        foreach (var archive in _archives)
        {
            var candidate = archive.Best;
            if (candidate is null) continue;
            if (best is null || Better(candidate.Evaluation, best.Evaluation)) best = candidate;
        }
        _best = best is null ? null : new BestSnapshot(Identity(best.Evaluation.GenomeId), best.Evaluation.Quality,
            best.Evaluation.Direction.ToString(), Identity(best.Evaluation.TaskVersionHash),
            Identity(best.Evaluation.EvaluatorVersionHash), Identity(best.Evaluation.ConfigurationHash));
    }

    private static bool Better(EvolutionEvaluation left, EvolutionEvaluation right)
    {
        int quality = Nullable.Compare(left.Quality, right.Quality);
        return quality == 0 ? string.CompareOrdinal(left.GenomeId, right.GenomeId) < 0
            : left.Direction == EvolutionOptimizationDirection.Maximize ? quality > 0 : quality < 0;
    }

    // Arbitrary adapter/operator labels may contain credentials. Preserve hash-shaped identities; hash other labels.
    private static string Identity(string value) => value.Length == 64 && value.All(char.IsAsciiHexDigit)
        ? value.ToLowerInvariant() : "sha256:" + EvolutionHash.Compute(value);

    internal Snapshot Read()
    {
        lock (_gate)
            return new Snapshot(_state, _finished, control.IsStopRequested, _events, _terminal, _attempts,
                _unknownCost ? null : _cost, _unknownCost, _pendingTruncated ? null : _pending.Count, null,
                "Only candidates seen in Proposed but not Evaluated; backend/in-flight queues are not exposed by this adapter.",
                _archiveCount, (int[])_islandCounts.Clone(), _best,
                "Archive-accepted feasible score, not an independent held-out correctness certificate.",
                new Dictionary<string, long>(_statuses, StringComparer.Ordinal),
                _lineage.Select(item => item with { Parents = (string[])item.Parents.Clone() }).ToArray(),
                _checkpointSequence, _checkpointHash, _checkpointHash is not null,
                "Model identity, live token usage and API currency are not exposed by this event contract; cost is current-segment evaluator units only.");
    }

    internal string Handle(string command)
    {
        bool cancel = false;
        lock (_gate)
        {
            if (!_finished && command == "pause")
            {
                if (control.RequestStop())
                    _state = "stop-requested"; // Not paused until the final committed checkpoint is verified.
            }
            else if (!_finished && command == "cancel")
            {
                _state = "cancel-requested";
                cancel = true;
            }
        }
        // Cancellation invokes caller callbacks synchronously; never invoke those while holding the snapshot lock.
        if (cancel) cancellation.Cancel();
        return JsonConvert.SerializeObject(Read());
    }

    internal async Task FinishAsync(EvolutionRunSummary? summary)
    {
        string? checkpointHash = null;
        if (summary?.CheckpointPath is { } path && _checkpointSequence is { } sequence && _checkpointEvent >= _lastEvaluationEvent)
        {
            try
            {
                var saved = await new JsonEvolutionCheckpointStore(path).LoadLatestAsync(summary.RunId).ConfigureAwait(false);
                if (saved is not null && saved.Sequence == sequence && saved.CompatibilityHash == summary.CompatibilityHash)
                {
                    using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
                    checkpointHash = Convert.ToHexString(SHA256.HashData(file)).ToLowerInvariant();
                }
            }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException or ArgumentException or InvalidOperationException)
            {
                // An old file's mere existence is not a resumability receipt.
            }
        }
        lock (_gate)
        {
            _finished = true;
            _checkpointHash = checkpointHash;
            _unknownCost |= summary is null; // No terminal receipt cannot be interpreted as zero abandoned work.
            _state = summary is null ? "aborted" : control.IsStopRequested &&
                summary.StopReason == EvolutionStopReason.Canceled && checkpointHash is not null ? "paused" : "stopped";
        }
    }
}
