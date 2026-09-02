using System.Collections.ObjectModel;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>Everything <see cref="EvolutionTraceFile"/> recovered from one trace file.</summary>
/// <remarks>
/// <para>
/// <see cref="Records"/> holds every record that parsed completely, in file order.
/// <see cref="IsComplete"/> distinguishes a trace that ends where its writer intended from one cut short by a crash,
/// a full disk, or a kill signal; in the latter case <see cref="Records"/> still contains every record that made it,
/// which is the point of writing traces incrementally. <see cref="Summary"/> is the sidecar metadata written beside
/// the trace, when it is present, and is the only place that can tell you a trace was deliberately truncated at a
/// configured bound rather than accidentally cut short.
/// </para>
/// <para>
/// OpenEvolve's loader has no equivalent: <c>load_traces_jsonl</c> calls <c>json.loads</c> on every line and raises on
/// the first partial one (trace_export_utils.py:204-210), so a trace from an interrupted run fails to load entirely
/// rather than yielding the records it does contain, and there is no metadata anywhere to consult about why.
/// </para>
/// <para><b>For Beginners:</b> When you read a run's diary back, this is what you get: the entries themselves, a flag
/// saying whether the diary ends properly or was cut off mid-sentence, and the cover page if one was saved. Check
/// <see cref="IsComplete"/> before drawing conclusions from a run you did not watch finish - if it is <c>false</c>,
/// the entries are still valid, there are just fewer of them than the run actually produced.</para>
/// </remarks>
public sealed class EvolutionTraceReadResult
{
    private readonly ReadOnlyCollection<EvolutionTraceRecord> _records;

    /// <summary>Initializes a read result.</summary>
    /// <param name="records">The records recovered, in file order.</param>
    /// <param name="isComplete">Whether the file ended where its writer intended.</param>
    /// <param name="format">The layout the file was written in.</param>
    /// <param name="compressed">Whether the file was gzip-compressed.</param>
    /// <param name="summary">The sidecar summary, when one was found.</param>
    /// <exception cref="ArgumentNullException"><paramref name="records"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="records"/> contains a <c>null</c> element.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="format"/> is undefined.</exception>
    public EvolutionTraceReadResult(IEnumerable<EvolutionTraceRecord> records, bool isComplete,
        EvolutionTraceFormat format, bool compressed, EvolutionTraceSummary? summary = null)
    {
        Guard.NotNull(records);
        if (!Enum.IsDefined(typeof(EvolutionTraceFormat), format)) throw new ArgumentOutOfRangeException(nameof(format));
        EvolutionTraceRecord[] copy = records.ToArray();
        foreach (EvolutionTraceRecord record in copy)
            if (record is null) throw new ArgumentException("A trace read result cannot contain null records.", nameof(records));
        _records = Array.AsReadOnly(copy);
        IsComplete = isComplete;
        Format = format;
        Compressed = compressed;
        Summary = summary;
    }

    /// <summary>Gets the recovered records in file order.</summary>
    public IReadOnlyList<EvolutionTraceRecord> Records => _records;

    /// <summary>Gets whether the file ended where its writer intended rather than being cut short.</summary>
    public bool IsComplete { get; }

    /// <summary>Gets the layout the file was written in, detected from its content.</summary>
    public EvolutionTraceFormat Format { get; }

    /// <summary>Gets whether the file was gzip-compressed, detected from its magic bytes.</summary>
    public bool Compressed { get; }

    /// <summary>Gets the sidecar summary written beside the trace, or <c>null</c> when none was found.</summary>
    public EvolutionTraceSummary? Summary { get; }
}
