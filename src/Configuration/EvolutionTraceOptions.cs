using System.Globalization;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Validation;

namespace AiDotNet.Configuration;

/// <summary>Configures the bounded, crash-safe evolution trace written by <see cref="EvolutionTraceObserver{TGenome}"/>.</summary>
/// <remarks>
/// <para>
/// Tracing is off by default and costs nothing until <see cref="Enabled"/> is set, at which point <see cref="Path"/>
/// becomes required: the observer refuses to guess a location, unlike OpenEvolve's tracer, which silently defaults to
/// <c>evolution_trace.jsonl</c> in the process working directory and then appends every subsequent run to that same
/// file, interleaving unrelated runs (evolution_trace.py:111-119 with <c>mode="a"</c> at trace_export_utils.py:172).
/// Use <see cref="EvolutionOutputLayout"/> to derive a per-run path from an output directory.
/// </para>
/// <para>
/// Two bounds make a trace safe to leave enabled for a long run. <see cref="MaxRecords"/> caps how many records are
/// written and <see cref="MaxBytes"/> caps their total serialized size; once either is reached the observer stops
/// writing, keeps counting what it dropped, and reports the truncation on its summary and in the trace's sidecar
/// metadata file. OpenEvolve has neither bound, and for its non-JSONL formats never clears its buffer at all
/// (evolution_trace.py:241-255), so a long run grows until the process runs out of memory. The three <c>Include</c>
/// flags trade detail for size: turning all three off roughly halves a record while keeping identity, status, quality,
/// cost, and the parent delta.
/// </para>
/// <para>
/// Tracing is an observer concern and never takes part in the engine's deterministic identity: no value here reaches
/// the configuration or compatibility hash, no value is checkpointed, and a failing or disabled tracer cannot change
/// which candidates a run produces. Two runs that differ only in these options therefore produce the same
/// <c>StateHash</c>.
/// </para>
/// <para><b>For Beginners:</b> A trace is a diary of everything the search tried, written as it happens so you can
/// study the run afterwards or feed it to another tool. Set <see cref="Enabled"/> and give it a <see cref="Path"/> and
/// you get one entry per evaluated candidate. <see cref="FlushEveryRecords"/> decides how often entries are pushed to
/// disk - smaller means less is lost if the machine dies, larger means fewer writes. <see cref="MaxRecords"/> and
/// <see cref="MaxBytes"/> are safety belts so an overnight run cannot fill your disk; if they trip, the trace stops
/// growing and tells you exactly how many entries it left out rather than failing the run. Turn on
/// <see cref="Compress"/> for long runs: traces are highly repetitive text and shrink dramatically.</para>
/// </remarks>
public sealed class EvolutionTraceOptions
{
    /// <summary>The largest value <see cref="FlushEveryRecords"/> may take.</summary>
    public const int MaximumFlushEveryRecords = 100_000;

    /// <summary>Gets or sets whether the observer writes anything at all; <c>false</c>, the default, disables tracing.</summary>
    public bool Enabled { get; set; }

    /// <summary>Gets or sets the trace file path; required when <see cref="Enabled"/> is set.</summary>
    /// <remarks>
    /// The observer creates the parent directory if needed and writes exactly this path. It does not append a
    /// <c>.gz</c> suffix for <see cref="Compress"/>; <see cref="EvolutionOutputLayout"/> supplies the conventional
    /// name, and the reader detects compression from the file's own magic bytes rather than from its extension.
    /// </remarks>
    public string? Path { get; set; }

    /// <summary>Gets or sets the on-disk layout; <see cref="EvolutionTraceFormat.JsonLines"/> by default.</summary>
    public EvolutionTraceFormat Format { get; set; } = EvolutionTraceFormat.JsonLines;

    /// <summary>Gets or sets whether the trace is gzip-compressed as a single stream.</summary>
    /// <remarks>
    /// The whole file is one gzip member, so records compress against each other. OpenEvolve reopens the file and
    /// starts a fresh gzip member for every single record (trace_export_utils.py:153-179), which adds a full header
    /// and trailer per record and prevents any cross-record compression.
    /// </remarks>
    public bool Compress { get; set; }

    /// <summary>Gets or sets how many buffered records trigger a write; the default of ten matches OpenEvolve.</summary>
    public int FlushEveryRecords { get; set; } = 10;

    /// <summary>Gets or sets the maximum total serialized record bytes; the default is 64 MiB.</summary>
    /// <remarks>
    /// The bound is measured on the UTF-8 encoded records before compression, so it limits content deterministically
    /// whatever the compression ratio turns out to be; with <see cref="Compress"/> set the file on disk is smaller.
    /// </remarks>
    public long MaxBytes { get; set; } = 64L * 1024L * 1024L;

    /// <summary>Gets or sets the maximum number of records written; the default is one million.</summary>
    public long MaxRecords { get; set; } = 1_000_000L;

    /// <summary>Gets or sets whether behaviour descriptors and the archive cell are recorded.</summary>
    public bool IncludeDescriptors { get; set; } = true;

    /// <summary>Gets or sets whether parents, inspirations, operator identifiers, and the seed stream are recorded.</summary>
    public bool IncludeLineage { get; set; } = true;

    /// <summary>Gets or sets whether diagnostic codes and messages are recorded.</summary>
    public bool IncludeDiagnostics { get; set; } = true;

    /// <summary>Gets or sets how many parent qualities are remembered so improvement deltas survive archive eviction.</summary>
    /// <remarks>
    /// OpenEvolve reads the parent straight out of the live database and simply skips the trace when the parent is
    /// gone (process_parallel.py:632-640), so exactly the traces describing a replaced parent - the interesting ones -
    /// are the ones it loses. This cache is bounded and evicts in insertion order; zero disables delta computation.
    /// </remarks>
    public int ParentQualityCacheSize { get; set; } = 4096;

    /// <summary>Validates every value and returns an independent copy.</summary>
    /// <returns>A defensive copy that later mutation of this instance cannot affect.</returns>
    /// <exception cref="ArgumentException"><see cref="Enabled"/> is set without a non-blank <see cref="Path"/>.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <see cref="Format"/> is undefined, <see cref="FlushEveryRecords"/> is outside one to
    /// <see cref="MaximumFlushEveryRecords"/>, <see cref="MaxBytes"/> or <see cref="MaxRecords"/> is not positive, or
    /// <see cref="ParentQualityCacheSize"/> is negative.
    /// </exception>
    internal EvolutionTraceOptions SnapshotAndValidate()
    {
        if (!Enum.IsDefined(typeof(EvolutionTraceFormat), Format)) throw new ArgumentOutOfRangeException(nameof(Format));
        if (Enabled && string.IsNullOrWhiteSpace(Path))
            throw new ArgumentException("An enabled evolution trace requires an explicit path.", nameof(Path));
        if (Path is not null && string.IsNullOrWhiteSpace(Path))
            throw new ArgumentException("The evolution trace path cannot be blank.", nameof(Path));
        if (FlushEveryRecords < 1 || FlushEveryRecords > MaximumFlushEveryRecords)
            throw new ArgumentOutOfRangeException(nameof(FlushEveryRecords),
                $"The flush interval must be between 1 and {MaximumFlushEveryRecords} records.");
        if (MaxBytes <= 0) throw new ArgumentOutOfRangeException(nameof(MaxBytes));
        if (MaxRecords <= 0) throw new ArgumentOutOfRangeException(nameof(MaxRecords));
        if (ParentQualityCacheSize < 0) throw new ArgumentOutOfRangeException(nameof(ParentQualityCacheSize));

        return new EvolutionTraceOptions
        {
            Enabled = Enabled,
            Path = Path?.Trim(),
            Format = Format,
            Compress = Compress,
            FlushEveryRecords = FlushEveryRecords,
            MaxBytes = MaxBytes,
            MaxRecords = MaxRecords,
            IncludeDescriptors = IncludeDescriptors,
            IncludeLineage = IncludeLineage,
            IncludeDiagnostics = IncludeDiagnostics,
            ParentQualityCacheSize = ParentQualityCacheSize
        };
    }

    /// <summary>Returns a stable, culture-independent representation of every value that changes what is recorded.</summary>
    /// <returns>The canonical text form, used for trace sidecar metadata rather than for engine identity.</returns>
    /// <remarks>
    /// The engine never consults this: tracing cannot change search results, so no trace option belongs in the
    /// configuration or compatibility hash. It is recorded alongside a trace so a later reader can tell which fields
    /// were omitted by configuration rather than missing from the run.
    /// </remarks>
    internal string ToCanonicalString() => string.Join("|", new[]
    {
        Enabled ? "on" : "off",
        ((int)Format).ToString(CultureInfo.InvariantCulture),
        Compress ? "gzip" : "plain",
        FlushEveryRecords.ToString(CultureInfo.InvariantCulture),
        MaxBytes.ToString(CultureInfo.InvariantCulture),
        MaxRecords.ToString(CultureInfo.InvariantCulture),
        IncludeDescriptors ? "descriptors" : "no-descriptors",
        IncludeLineage ? "lineage" : "no-lineage",
        IncludeDiagnostics ? "diagnostics" : "no-diagnostics",
        ParentQualityCacheSize.ToString(CultureInfo.InvariantCulture)
    });

    /// <summary>Returns the resolved trace path, throwing when tracing is enabled without one.</summary>
    /// <returns>The non-blank configured path.</returns>
    /// <exception cref="InvalidOperationException">No path is configured.</exception>
    internal string RequirePath()
    {
        if (Path is null || string.IsNullOrWhiteSpace(Path))
            throw new InvalidOperationException("An enabled evolution trace requires an explicit path.");
        Guard.NotNullOrWhiteSpace(Path);
        return Path;
    }
}
