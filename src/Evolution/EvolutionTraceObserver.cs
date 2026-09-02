using System.Globalization;
using System.IO.Compression;
using System.Text;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Streams one bounded, crash-safe trace record per evaluation to a file while a run is in flight.</summary>
/// <typeparam name="TGenome">The task-specific genome type.</typeparam>
/// <remarks>
/// <para>
/// Attach this as the engine's observer and every evaluation that reaches a terminal status appends one
/// <see cref="EvolutionTraceRecord"/> to <see cref="TracePath"/>. Records are buffered and written every
/// <c>EvolutionTraceOptions.FlushEveryRecords</c>, and pushed to the operating system in full whenever the engine
/// publishes a checkpoint, when the run stops, and when this observer is disposed - so a trace and the checkpoint
/// beside it always describe the same moment. OpenEvolve's tracer is never flushed on a checkpoint at all
/// (controller.py:142-163, 354-363): it flushes only when its own buffer fills and when the run closes cleanly.
/// </para>
/// <para>
/// It cannot break a run. Every file operation is guarded, and a failure - a full disk, a revoked permission, a
/// deleted directory - marks the trace faulted, counts the failure on <see cref="Summary"/>, and stops writing rather
/// than propagating. The engine isolates observer exceptions as well, but a tracer that silently disables itself and
/// then reports how much it lost is more useful than one that is retried thousands of times against a disk that is
/// still full. It also cannot grow without bound: <c>MaxRecords</c> and <c>MaxBytes</c> stop the trace and set
/// <c>IsTruncated</c> on the summary, where OpenEvolve has no bound of any kind and, for its non-JSONL formats, never
/// even clears its in-memory buffer (evolution_trace.py:241-255).
/// </para>
/// <para>
/// Two coverage differences over OpenEvolve are worth knowing. It writes a record for every terminal status - a
/// duplicate, a novelty rejection, a timeout, a failure - where OpenEvolve traces only children that evaluated
/// successfully (process_parallel.py:613-634). And it keeps its own bounded cache of parent qualities, so the
/// improvement delta survives the parent being replaced or evicted; OpenEvolve reads the parent from the live
/// database and skips the trace entirely when it is gone (process_parallel.py:632-640), losing exactly the traces
/// that describe a beaten parent.
/// </para>
/// <para>
/// Tracing never touches the engine's deterministic identity: no trace option reaches the configuration or
/// compatibility hash, nothing is checkpointed, and two runs differing only in their trace settings produce the same
/// state hash.
/// </para>
/// <para><b>For Beginners:</b> This writes the run's diary. Create one with a path, hand it to the engine as its
/// observer, and dispose it when the run is done - a <c>using</c> statement does that for you and makes sure the last
/// entries reach disk. Afterwards, load the file with <see cref="EvolutionTraceFile"/> to see what the search tried
/// and in what order. While the run is going you can read <see cref="Summary"/> for a live count of entries written
/// and how often candidates beat their parents. If you left the trace enabled overnight and it hit its size limit,
/// the summary tells you exactly how many entries were left out.</para>
/// </remarks>
public sealed class EvolutionTraceObserver<TGenome> : IEvolutionObserver<TGenome>, IDisposable
{
    private static readonly IReadOnlyDictionary<string, double> EmptyValues =
        new System.Collections.ObjectModel.ReadOnlyDictionary<string, double>(
            new Dictionary<string, double>(StringComparer.Ordinal));
    private static readonly IReadOnlyList<string> EmptyIdentities = Array.AsReadOnly(Array.Empty<string>());
    private static readonly IReadOnlyList<EvolutionDiagnostic> EmptyDiagnostics =
        Array.AsReadOnly(Array.Empty<EvolutionDiagnostic>());

    private readonly object _gate = new();
    private readonly EvolutionTraceOptions _options;
    private readonly IReadOnlyList<EvolutionDescriptorDefinition> _descriptors;
    private readonly List<string> _buffer = new();
    private readonly Dictionary<string, ParentSnapshot> _parents = new(StringComparer.Ordinal);
    private readonly Queue<string> _parentOrder = new();
    private readonly Dictionary<EvolutionEvaluationStatus, long> _statusCounts = new();
    private readonly Dictionary<string, double> _totalMetricDeltas = new(StringComparer.Ordinal);
    private readonly Dictionary<string, double> _bestMetricDeltas = new(StringComparer.Ordinal);
    private readonly Dictionary<string, double> _worstMetricDeltas = new(StringComparer.Ordinal);
    private readonly string _runId;

    private FileStream? _file;
    private Stream? _compression;
    private StreamWriter? _writer;
    private bool _wroteAnyRecord;
    private bool _faulted;
    private bool _truncated;
    private bool _closed;
    private bool _disposed;
    private long _recordsWritten;
    private long _recordsDropped;
    private long _bytesWritten;
    private long _pendingBytes;
    private long _observerFailures;
    private long _improvementCount;
    private long _deltaSampleCount;
    private long _cacheHits;
    private double _totalCostUnits;
    private long _totalElapsedTicks;
    private double? _bestQualityDelta;
    private double? _worstQualityDelta;
    private string? _firstFailureMessage;

    /// <summary>Initializes a trace observer.</summary>
    /// <param name="options">The trace configuration; validated and copied, so later mutation cannot affect the run.</param>
    /// <param name="runId">The non-blank run identifier stamped on the trace summary.</param>
    /// <param name="descriptors">
    /// The archive's descriptor definitions, used to record each record's archive cell. Pass <c>null</c> - or leave
    /// <c>EvolutionTraceOptions.IncludeDescriptors</c> clear - to omit the cell.
    /// </param>
    /// <exception cref="ArgumentNullException"><paramref name="options"/> or <paramref name="runId"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="runId"/> is blank, <paramref name="descriptors"/> contains a <c>null</c> element, or the
    /// options are enabled without a path.
    /// </exception>
    public EvolutionTraceObserver(EvolutionTraceOptions options, string runId,
        IReadOnlyList<EvolutionDescriptorDefinition>? descriptors = null)
    {
        Guard.NotNull(options);
        Guard.NotNullOrWhiteSpace(runId);
        _options = options.SnapshotAndValidate();
        _runId = runId.Trim();
        EvolutionDescriptorDefinition[] copy = descriptors?.ToArray() ?? Array.Empty<EvolutionDescriptorDefinition>();
        foreach (EvolutionDescriptorDefinition descriptor in copy)
            if (descriptor is null)
                throw new ArgumentException("Descriptor definitions cannot contain null values.", nameof(descriptors));
        _descriptors = Array.AsReadOnly(copy);
        TracePath = _options.Enabled ? Path.GetFullPath(_options.RequirePath()) : string.Empty;
        SummaryPath = _options.Enabled ? EvolutionOutputLayout.SummaryPathFor(TracePath) : string.Empty;
    }

    /// <summary>Gets the resolved trace file path, or an empty string when tracing is disabled.</summary>
    public string TracePath { get; }

    /// <summary>Gets the resolved sidecar summary path, or an empty string when tracing is disabled.</summary>
    public string SummaryPath { get; }

    /// <summary>Gets whether the observer is writing anything.</summary>
    public bool IsEnabled => _options.Enabled;

    /// <summary>Gets a live snapshot of what has been written and what the run looks like so far.</summary>
    public EvolutionTraceSummary Summary
    {
        get { lock (_gate) return CreateSummary(); }
    }

    /// <inheritdoc/>
    public ValueTask OnEventAsync(EvolutionEvent<TGenome> evolutionEvent, CancellationToken cancellationToken = default)
    {
        if (evolutionEvent is null || !_options.Enabled) return default;
        lock (_gate)
        {
            if (_disposed) return default;
            switch (evolutionEvent.Kind)
            {
                case EvolutionEventKind.Evaluated:
                    if (evolutionEvent.Candidate is not null && evolutionEvent.Evaluation is not null)
                    {
                        Append(BuildRecord(evolutionEvent.Sequence, evolutionEvent.Candidate, evolutionEvent.Evaluation,
                            evolutionEvent.InsertionResult));
                    }
                    break;
                case EvolutionEventKind.Checkpointed:
                    FlushDurable();
                    break;
                case EvolutionEventKind.Stopped:
                    Close();
                    break;
                default:
                    break;
            }
        }
        return default;
    }

    /// <summary>Flushes every buffered record and refreshes the sidecar summary.</summary>
    /// <remarks>Safe to call at any time and from any thread; failures are counted, never thrown.</remarks>
    public void Flush()
    {
        if (!_options.Enabled) return;
        lock (_gate) FlushDurable();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            if (_options.Enabled) Close();
            _disposed = true;
        }
    }

    private EvolutionTraceRecord BuildRecord(long sequence, EvolutionCandidate<TGenome> candidate,
        EvolutionEvaluation evaluation, EvolutionArchiveInsertionResult? insertion)
    {
        EvolutionLineage lineage = candidate.Lineage;
        ParentSnapshot? parent = FindParent(lineage);
        double? qualityDelta = null;
        bool isImprovement = false;
        var metricDeltas = new Dictionary<string, double>(StringComparer.Ordinal);
        if (parent is not null && parent.Quality.HasValue && evaluation.Quality.HasValue)
        {
            double delta = evaluation.Quality.Value - parent.Quality.Value;
            if (EvolutionDescriptorDefinition.IsFinite(delta))
            {
                qualityDelta = delta;
                isImprovement = evaluation.Direction == EvolutionOptimizationDirection.Maximize ? delta > 0 : delta < 0;
            }
        }
        if (parent is not null)
        {
            foreach (KeyValuePair<string, double> metric in evaluation.Metrics)
            {
                if (!parent.Metrics.TryGetValue(metric.Key, out double parentValue)) continue;
                double delta = metric.Value - parentValue;
                if (EvolutionDescriptorDefinition.IsFinite(delta)) metricDeltas[metric.Key] = delta;
            }
        }

        var record = new EvolutionTraceRecord(sequence, evaluation.EvaluationId, evaluation.GenomeId,
            evaluation.Status, evaluation.Direction, lineage.Island, lineage.Generation, evaluation.CacheStatus,
            DateTimeOffset.UtcNow, evaluation.TaskVersionHash, evaluation.EvaluatorVersionHash,
            evaluation.ConfigurationHash)
        {
            Quality = evaluation.Quality,
            InsertionResult = insertion,
            Cell = _options.IncludeDescriptors ? TryCreateCell(evaluation.Descriptors) : null,
            Descriptors = _options.IncludeDescriptors ? evaluation.Descriptors : EmptyValues,
            Metrics = evaluation.Metrics,
            ParentGenomeId = parent?.GenomeId,
            ParentQuality = parent?.Quality,
            QualityDelta = qualityDelta,
            IsImprovement = isImprovement,
            MetricDeltas = metricDeltas,
            ParentIds = _options.IncludeLineage ? lineage.ParentIds : EmptyIdentities,
            InspirationIds = _options.IncludeLineage ? lineage.InspirationIds : EmptyIdentities,
            VariationOperatorId = _options.IncludeLineage ? lineage.VariationOperatorId : null,
            RefinerId = _options.IncludeLineage ? lineage.RefinerId : null,
            SeedStream = _options.IncludeLineage ? lineage.SeedStream : 0UL,
            AttemptCount = evaluation.Cost.AttemptCount,
            CostUnits = evaluation.Cost.CostUnits,
            Elapsed = evaluation.Cost.Elapsed,
            RejectedStage = evaluation.Cost.RejectedStage,
            Diagnostics = _options.IncludeDiagnostics ? evaluation.Diagnostics : EmptyDiagnostics
        };

        RememberParent(evaluation);
        return record;
    }

    private ParentSnapshot? FindParent(EvolutionLineage lineage)
    {
        foreach (string parentId in lineage.ParentIds)
            if (_parents.TryGetValue(parentId, out ParentSnapshot? snapshot)) return snapshot;
        return null;
    }

    private void RememberParent(EvolutionEvaluation evaluation)
    {
        if (_options.ParentQualityCacheSize == 0) return;
        if (evaluation.Status != EvolutionEvaluationStatus.Completed || !evaluation.Quality.HasValue) return;
        if (!_parents.ContainsKey(evaluation.GenomeId)) _parentOrder.Enqueue(evaluation.GenomeId);
        _parents[evaluation.GenomeId] = new ParentSnapshot(evaluation.GenomeId, evaluation.Quality,
            new Dictionary<string, double>(evaluation.Metrics.ToDictionary(item => item.Key, item => item.Value),
                StringComparer.Ordinal));
        while (_parentOrder.Count > _options.ParentQualityCacheSize)
        {
            string evicted = _parentOrder.Dequeue();
            _parents.Remove(evicted);
        }
    }

    private string? TryCreateCell(IReadOnlyDictionary<string, double> descriptors)
    {
        if (_descriptors.Count == 0) return null;
        var bins = new int[_descriptors.Count];
        for (int i = 0; i < _descriptors.Count; i++)
        {
            if (!descriptors.TryGetValue(_descriptors[i].Name, out double value) ||
                !_descriptors[i].TryGetBin(value, out bins[i]))
            {
                return null;
            }
        }
        return new EvolutionCellKey(bins).StableKey;
    }

    private void Append(EvolutionTraceRecord record)
    {
        if (_faulted || _truncated || _closed)
        {
            _recordsDropped++;
            return;
        }

        string line;
        long lineBytes;
        try
        {
            line = EvolutionTraceFile.SerializeRecord(record);
            lineBytes = Encoding.UTF8.GetByteCount(line) + 1;
        }
        catch (Exception exception) when (IsRecoverable(exception))
        {
            RecordFailure(exception);
            _recordsDropped++;
            return;
        }

        if (_recordsWritten + _buffer.Count >= _options.MaxRecords ||
            _bytesWritten + _pendingBytes + lineBytes > _options.MaxBytes)
        {
            _truncated = true;
            _recordsDropped++;
            return;
        }

        _buffer.Add(line);
        _pendingBytes += lineBytes;
        AccumulateStatistics(record);
        if (_buffer.Count >= _options.FlushEveryRecords) FlushRecords();
    }

    private void AccumulateStatistics(EvolutionTraceRecord record)
    {
        _statusCounts.TryGetValue(record.Status, out long count);
        _statusCounts[record.Status] = count + 1;
        if (record.CacheStatus == EvolutionCacheStatus.Hit) _cacheHits++;
        _totalCostUnits = SaturatingAdd(_totalCostUnits, record.CostUnits);
        _totalElapsedTicks = SaturatingAdd(_totalElapsedTicks, record.Elapsed.Ticks);

        if (record.QualityDelta.HasValue)
        {
            _deltaSampleCount++;
            if (record.IsImprovement) _improvementCount++;
            double delta = record.QualityDelta.Value;
            _bestQualityDelta = _bestQualityDelta.HasValue ? Math.Max(_bestQualityDelta.Value, delta) : delta;
            _worstQualityDelta = _worstQualityDelta.HasValue ? Math.Min(_worstQualityDelta.Value, delta) : delta;
        }

        foreach (KeyValuePair<string, double> pair in record.MetricDeltas)
        {
            _totalMetricDeltas.TryGetValue(pair.Key, out double total);
            _totalMetricDeltas[pair.Key] = SaturatingAdd(total, pair.Value);
            _bestMetricDeltas[pair.Key] = _bestMetricDeltas.TryGetValue(pair.Key, out double best)
                ? Math.Max(best, pair.Value)
                : pair.Value;
            _worstMetricDeltas[pair.Key] = _worstMetricDeltas.TryGetValue(pair.Key, out double worst)
                ? Math.Min(worst, pair.Value)
                : pair.Value;
        }
    }

    private void FlushRecords()
    {
        if (_buffer.Count == 0 || _faulted || _closed) return;
        try
        {
            EnsureOpen();
            StreamWriter writer = _writer ?? throw new InvalidOperationException("The trace writer is not open.");
            foreach (string line in _buffer)
            {
                if (_options.Format == EvolutionTraceFormat.Json)
                {
                    if (_wroteAnyRecord) writer.Write(",\n");
                    writer.Write(line);
                }
                else
                {
                    writer.Write(line);
                    writer.Write('\n');
                }
                _wroteAnyRecord = true;
                _recordsWritten++;
            }
            writer.Flush();
            _compression?.Flush();
            _file?.Flush();
            _bytesWritten += _pendingBytes;
        }
        catch (Exception exception) when (IsRecoverable(exception))
        {
            RecordFailure(exception);
            _recordsDropped += _buffer.Count;
        }
        finally
        {
            _buffer.Clear();
            _pendingBytes = 0;
        }
    }

    private void FlushDurable()
    {
        FlushRecords();
        WriteSummaryFile();
    }

    private void Close()
    {
        if (_closed) return;
        FlushRecords();
        _closed = true;
        try
        {
            // An enabled trace always leaves a readable file, even when the run produced no records at all.
            if (!_faulted) EnsureOpen();
            if (_writer is not null && _options.Format == EvolutionTraceFormat.Json)
            {
                _writer.Write(EvolutionTraceFile.JsonFooter(CreateSummary()));
                _writer.Flush();
            }
        }
        catch (Exception exception) when (IsRecoverable(exception))
        {
            RecordFailure(exception);
        }

        DisposeWriter();
        WriteSummaryFile();
    }

    private void EnsureOpen()
    {
        if (_writer is not null) return;
        string directory = Path.GetDirectoryName(TracePath) ?? ".";
        if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);
        _file = new FileStream(TracePath, FileMode.Create, FileAccess.Write, FileShare.Read);
        _compression = _options.Compress ? new GZipStream(_file, CompressionMode.Compress, leaveOpen: true) : null;
        Stream target = _compression ?? (Stream)_file;
        _writer = new StreamWriter(target, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false), 4096,
            leaveOpen: true);
        if (_options.Format == EvolutionTraceFormat.Json) _writer.Write(EvolutionTraceFile.JsonHeader(_runId));
    }

    private void DisposeWriter()
    {
        try { _writer?.Dispose(); }
        catch (Exception exception) when (IsRecoverable(exception)) { RecordFailure(exception); }
        try { _compression?.Dispose(); }
        catch (Exception exception) when (IsRecoverable(exception)) { RecordFailure(exception); }
        try { _file?.Dispose(); }
        catch (Exception exception) when (IsRecoverable(exception)) { RecordFailure(exception); }
        _writer = null;
        _compression = null;
        _file = null;
    }

    private void WriteSummaryFile()
    {
        try
        {
            EvolutionTraceFile.WriteSummary(SummaryPath, CreateSummary());
        }
        catch (Exception exception) when (IsRecoverable(exception))
        {
            RecordFailure(exception);
        }
    }

    private EvolutionTraceSummary CreateSummary() => new(_runId, _options.Format, _options.Compress)
    {
        TraceOptions = _options.ToCanonicalString(),
        RecordsWritten = _recordsWritten,
        RecordsDropped = _recordsDropped,
        BytesWritten = _bytesWritten,
        IsTruncated = _truncated || _faulted,
        IsClosed = _closed,
        ObserverFailures = _observerFailures,
        FirstFailureMessage = _firstFailureMessage,
        ImprovementCount = _improvementCount,
        DeltaSampleCount = _deltaSampleCount,
        CacheHits = _cacheHits,
        BestQualityDelta = _bestQualityDelta,
        WorstQualityDelta = _worstQualityDelta,
        TotalCostUnits = _totalCostUnits,
        TotalElapsed = TimeSpan.FromTicks(_totalElapsedTicks),
        StatusCounts = _statusCounts,
        TotalMetricDeltas = _totalMetricDeltas,
        BestMetricDeltas = _bestMetricDeltas,
        WorstMetricDeltas = _worstMetricDeltas
    };

    private void RecordFailure(Exception exception)
    {
        _faulted = true;
        _observerFailures++;
        if (_firstFailureMessage is null)
        {
            _firstFailureMessage = string.Format(CultureInfo.InvariantCulture, "{0}: {1}",
                exception.GetType().Name, exception.Message);
        }
        DisposeWriterQuietly();
    }

    private void DisposeWriterQuietly()
    {
        // Already failing: a second failure while closing adds nothing the first failure has not already reported.
        try { _writer?.Dispose(); }
        catch (Exception exception) when (IsRecoverable(exception)) { _observerFailures++; }
        try { _compression?.Dispose(); }
        catch (Exception exception) when (IsRecoverable(exception)) { _observerFailures++; }
        try { _file?.Dispose(); }
        catch (Exception exception) when (IsRecoverable(exception)) { _observerFailures++; }
        _writer = null;
        _compression = null;
        _file = null;
    }

    private static bool IsRecoverable(Exception exception) => exception is IOException or UnauthorizedAccessException or
        ObjectDisposedException or NotSupportedException or System.Security.SecurityException or ArgumentException or
        InvalidOperationException or Newtonsoft.Json.JsonException;

    private static double SaturatingAdd(double left, double right)
    {
        double sum = left + right;
        return EvolutionDescriptorDefinition.IsFinite(sum) ? sum : double.MaxValue;
    }

    private static long SaturatingAdd(long left, long right)
    {
        long sum = unchecked(left + right);
        return left > 0 && right > 0 && sum < 0 ? long.MaxValue : sum;
    }

    private sealed class ParentSnapshot
    {
        public ParentSnapshot(string genomeId, double? quality, IReadOnlyDictionary<string, double> metrics)
        {
            GenomeId = genomeId;
            Quality = quality;
            Metrics = metrics;
        }

        public string GenomeId { get; }
        public double? Quality { get; }
        public IReadOnlyDictionary<string, double> Metrics { get; }
    }
}
