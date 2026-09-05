using System.Collections.ObjectModel;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Accounting and statistics for one evolution trace, kept live by the observer and stored beside the file.</summary>
/// <remarks>
/// <para>
/// The observer maintains one of these while a run is in flight and writes it to a sidecar metadata file next to the
/// trace on every durable flush, so an interrupted run still leaves an accurate account of what its trace contains.
/// It answers two different questions. The accounting half - <see cref="RecordsWritten"/>,
/// <see cref="RecordsDropped"/>, <see cref="BytesWritten"/>, <see cref="IsTruncated"/>, <see cref="IsClosed"/>,
/// <see cref="ObserverFailures"/> - says how much of the run reached the file and why anything is missing. The
/// statistics half - <see cref="ImprovementCount"/>, <see cref="ImprovementRate"/>, the per-metric delta maps,
/// <see cref="StatusCounts"/>, <see cref="CacheHitRate"/>, <see cref="MeanEvaluatorSeconds"/> - summarises the run
/// itself.
/// </para>
/// <para>
/// OpenEvolve keeps a comparable statistics dictionary but only logs it to the console at close
/// (evolution_trace.py:210-233, 301-309): it is never persisted, never exposed to the caller, and lost entirely if
/// the process dies. It also counts an improvement as any positive delta on a metric literally named
/// <c>combined_score</c>, which scores every minimising objective backwards; <see cref="ImprovementCount"/> here is
/// direction-aware. And it has no notion of truncation, dropped records, or write failures, because it has no bounds
/// and swallows its write errors without counting them (evolution_trace.py:207-208, 257-258).
/// </para>
/// <para><b>For Beginners:</b> This is the cover page of the run's diary. It tells you how many entries were written,
/// whether any were left out because a size limit was hit, and a quick read on how the search went: what fraction of
/// candidates actually beat their parent, how often the cache saved a real evaluation, how long an evaluation took on
/// average, and how many candidates ended in each outcome. Because it is saved next to the trace file, you get all of
/// that even if the run was killed halfway through.</para>
/// </remarks>
public sealed class EvolutionTraceSummary
{
    /// <summary>The schema version stamped on summaries written by this build.</summary>
    public const int CurrentSchemaVersion = 1;

    private readonly ReadOnlyDictionary<EvolutionEvaluationStatus, long> _statusCounts =
        new(new Dictionary<EvolutionEvaluationStatus, long>());
    private readonly ReadOnlyDictionary<string, double> _totalMetricDeltas =
        new(new Dictionary<string, double>(StringComparer.Ordinal));
    private readonly ReadOnlyDictionary<string, double> _bestMetricDeltas =
        new(new Dictionary<string, double>(StringComparer.Ordinal));
    private readonly ReadOnlyDictionary<string, double> _worstMetricDeltas =
        new(new Dictionary<string, double>(StringComparer.Ordinal));
    private readonly long _recordsWritten;
    private readonly long _recordsDropped;
    private readonly long _bytesWritten;
    private readonly long _observerFailures;
    private readonly long _improvementCount;
    private readonly long _deltaSampleCount;
    private readonly long _cacheHits;
    private readonly double _totalCostUnits;
    private readonly TimeSpan _totalElapsed;

    /// <summary>Initializes a summary for one trace file.</summary>
    /// <param name="runId">The non-blank run identifier the trace belongs to.</param>
    /// <param name="format">The on-disk layout the trace was written in.</param>
    /// <param name="compressed">Whether the trace is gzip-compressed.</param>
    /// <exception cref="ArgumentNullException"><paramref name="runId"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="runId"/> is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="format"/> is undefined.</exception>
    public EvolutionTraceSummary(string runId, EvolutionTraceFormat format, bool compressed)
    {
        Guard.NotNullOrWhiteSpace(runId);
        if (!Enum.IsDefined(typeof(EvolutionTraceFormat), format)) throw new ArgumentOutOfRangeException(nameof(format));
        RunId = runId.Trim();
        Format = format;
        Compressed = compressed;
    }

    /// <summary>Gets the schema version this summary conforms to.</summary>
    public int SchemaVersion => CurrentSchemaVersion;

    /// <summary>Gets the run identifier the trace belongs to.</summary>
    public string RunId { get; }

    /// <summary>Gets the on-disk layout the trace was written in.</summary>
    public EvolutionTraceFormat Format { get; }

    /// <summary>Gets whether the trace is gzip-compressed.</summary>
    public bool Compressed { get; }

    /// <summary>Gets the canonical form of the trace options in effect, so a reader knows which fields were omitted.</summary>
    public string TraceOptions { get; init; } = string.Empty;

    /// <summary>Gets how many records reached the file.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The assigned value is negative.</exception>
    public long RecordsWritten
    {
        get => _recordsWritten;
        init => _recordsWritten = NonNegative(value);
    }

    /// <summary>Gets how many records were produced but not written, because a bound or a write failure stopped them.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The assigned value is negative.</exception>
    public long RecordsDropped
    {
        get => _recordsDropped;
        init => _recordsDropped = NonNegative(value);
    }

    /// <summary>Gets the total uncompressed serialized bytes of the records written.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The assigned value is negative.</exception>
    public long BytesWritten
    {
        get => _bytesWritten;
        init => _bytesWritten = NonNegative(value);
    }

    /// <summary>Gets whether a record bound or a write failure stopped the trace before the run ended.</summary>
    public bool IsTruncated { get; init; }

    /// <summary>Gets whether the trace was closed cleanly rather than left open by a crash.</summary>
    public bool IsClosed { get; init; }

    /// <summary>Gets how many observer operations failed and were swallowed rather than propagated into the run.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The assigned value is negative.</exception>
    public long ObserverFailures
    {
        get => _observerFailures;
        init => _observerFailures = NonNegative(value);
    }

    /// <summary>Gets the message of the first swallowed observer failure, when one occurred.</summary>
    public string? FirstFailureMessage { get; init; }

    /// <summary>Gets how many candidates improved on their parent in the run's optimization direction.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The assigned value is negative.</exception>
    public long ImprovementCount
    {
        get => _improvementCount;
        init => _improvementCount = NonNegative(value);
    }

    /// <summary>Gets how many records carried a comparable parent quality, the denominator of <see cref="ImprovementRate"/>.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The assigned value is negative.</exception>
    public long DeltaSampleCount
    {
        get => _deltaSampleCount;
        init => _deltaSampleCount = NonNegative(value);
    }

    /// <summary>Gets the fraction of comparable candidates that beat their parent, or zero when none were comparable.</summary>
    public double ImprovementRate => _deltaSampleCount == 0 ? 0 : (double)_improvementCount / _deltaSampleCount;

    /// <summary>Gets how many records were served from the evaluation cache.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The assigned value is negative.</exception>
    public long CacheHits
    {
        get => _cacheHits;
        init => _cacheHits = NonNegative(value);
    }

    /// <summary>Gets the fraction of written records that were cache hits, or zero when none were written.</summary>
    public double CacheHitRate => _recordsWritten == 0 ? 0 : (double)_cacheHits / _recordsWritten;

    /// <summary>Gets the best quality delta observed, in the run's optimization direction, or <c>null</c> when none was.</summary>
    public double? BestQualityDelta { get; init; }

    /// <summary>Gets the worst quality delta observed, in the run's optimization direction, or <c>null</c> when none was.</summary>
    public double? WorstQualityDelta { get; init; }

    /// <summary>Gets the sum of every metric delta observed, keyed by metric name.</summary>
    public IReadOnlyDictionary<string, double> TotalMetricDeltas
    {
        get => _totalMetricDeltas;
        init => _totalMetricDeltas = CopyDeltas(value);
    }

    /// <summary>Gets the largest delta observed for each metric.</summary>
    public IReadOnlyDictionary<string, double> BestMetricDeltas
    {
        get => _bestMetricDeltas;
        init => _bestMetricDeltas = CopyDeltas(value);
    }

    /// <summary>Gets the smallest delta observed for each metric.</summary>
    public IReadOnlyDictionary<string, double> WorstMetricDeltas
    {
        get => _worstMetricDeltas;
        init => _worstMetricDeltas = CopyDeltas(value);
    }

    /// <summary>Gets how many records ended in each terminal status.</summary>
    public IReadOnlyDictionary<EvolutionEvaluationStatus, long> StatusCounts
    {
        get => _statusCounts;
        init
        {
            var copy = new Dictionary<EvolutionEvaluationStatus, long>();
            if (value is not null)
            {
                foreach (KeyValuePair<EvolutionEvaluationStatus, long> pair in value.OrderBy(item => (int)item.Key))
                {
                    if (!Enum.IsDefined(typeof(EvolutionEvaluationStatus), pair.Key))
                        throw new ArgumentOutOfRangeException(nameof(value), "Status counts contain an undefined status.");
                    if (pair.Value < 0) throw new ArgumentOutOfRangeException(nameof(value), "Status counts cannot be negative.");
                    copy[pair.Key] = pair.Value;
                }
            }
            _statusCounts = new ReadOnlyDictionary<EvolutionEvaluationStatus, long>(copy);
        }
    }

    /// <summary>Gets the sum of task-defined resource units across every record written.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The assigned value is negative or not finite.</exception>
    public double TotalCostUnits
    {
        get => _totalCostUnits;
        init
        {
            if (!EvolutionDescriptorDefinition.IsFinite(value) || value < 0)
                throw new ArgumentOutOfRangeException(nameof(value), "Total cost units must be finite and non-negative.");
            _totalCostUnits = value;
        }
    }

    /// <summary>Gets the summed evaluator wall-clock time across every record written.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The assigned value is negative.</exception>
    public TimeSpan TotalElapsed
    {
        get => _totalElapsed;
        init
        {
            if (value < TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(value));
            _totalElapsed = value;
        }
    }

    /// <summary>Gets the mean evaluator seconds per written record, or zero when none were written.</summary>
    public double MeanEvaluatorSeconds => _recordsWritten == 0 ? 0 : _totalElapsed.TotalSeconds / _recordsWritten;

    private static long NonNegative(long value)
    {
        if (value < 0) throw new ArgumentOutOfRangeException(nameof(value), "Trace counters cannot be negative.");
        return value;
    }

    private static ReadOnlyDictionary<string, double> CopyDeltas(IReadOnlyDictionary<string, double>? values)
    {
        var copy = new Dictionary<string, double>(StringComparer.Ordinal);
        if (values is null) return new ReadOnlyDictionary<string, double>(copy);
        foreach (KeyValuePair<string, double> pair in values.OrderBy(item => item.Key, StringComparer.Ordinal))
        {
            if (string.IsNullOrWhiteSpace(pair.Key))
                throw new ArgumentException("Metric names cannot be empty or white space.", nameof(values));
            if (!EvolutionDescriptorDefinition.IsFinite(pair.Value))
                throw new ArgumentException("Metric deltas must be finite.", nameof(values));
            copy[pair.Key.Trim()] = pair.Value;
        }
        return new ReadOnlyDictionary<string, double>(copy);
    }
}
