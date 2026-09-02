using System.Collections.ObjectModel;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>An in-memory <see cref="IReasoningModelDiagnosticSink"/> that keeps the most recent adjustments.</summary>
/// <remarks>
/// <para>
/// A long evolution run makes the same handful of adjustments on every single request, so an unbounded list would
/// grow without adding information. This sink keeps a bounded ring of the most recent records for reading back in
/// order, and separately keeps one representative record and a count for each distinct adjustment, so a run of a
/// million calls still summarises to a few lines. Both views are safe to read while requests are in flight.
/// </para>
/// <para>
/// Use it in tests to assert exactly which settings a profile removed, and in short production runs to print a
/// summary at the end. For a long-lived service, forward to the host's own logging instead by implementing
/// <see cref="IReasoningModelDiagnosticSink"/> directly.
/// </para>
/// <para><b>For Beginners:</b> Hand one of these to a reasoning-aware chat client and it collects the notices about
/// settings that had to be changed. Afterwards, <see cref="GetSummary"/> tells you each distinct change and how
/// many times it happened — usually one line saying "temperature was dropped, 400 times", which is exactly what you
/// want to know.</para>
/// </remarks>
public sealed class CollectingReasoningModelDiagnosticSink : IReasoningModelDiagnosticSink
{
    /// <summary>The number of recent records retained when no capacity is supplied.</summary>
    public const int DefaultCapacity = 256;

    private readonly object _gate = new();
    private readonly Queue<ReasoningModelDiagnostic> _recent = new();
    private readonly Dictionary<string, ReasoningModelDiagnostic> _distinct =
        new(StringComparer.Ordinal);
    private readonly Dictionary<string, long> _counts = new(StringComparer.Ordinal);
    private readonly int _capacity;
    private long _total;

    /// <summary>Initializes a collecting sink.</summary>
    /// <param name="capacity">How many recent records to retain; defaults to <see cref="DefaultCapacity"/>.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="capacity"/> is not positive.</exception>
    public CollectingReasoningModelDiagnosticSink(int capacity = DefaultCapacity)
    {
        Guard.Positive(capacity);
        _capacity = capacity;
    }

    /// <summary>Gets how many adjustments have been reported since construction.</summary>
    public long TotalReported
    {
        get
        {
            lock (_gate) return _total;
        }
    }

    /// <inheritdoc/>
    public void Report(ReasoningModelDiagnostic diagnostic)
    {
        Guard.NotNull(diagnostic);
        lock (_gate)
        {
            _total++;
            _recent.Enqueue(diagnostic);
            while (_recent.Count > _capacity) _recent.Dequeue();

            string key = diagnostic.Key;
            if (_counts.TryGetValue(key, out long count))
            {
                _counts[key] = count + 1L;
            }
            else
            {
                _counts[key] = 1L;
                _distinct[key] = diagnostic;
            }
        }
    }

    /// <summary>Gets the most recent adjustments, oldest first.</summary>
    /// <returns>Up to the configured capacity of records; empty when nothing was reported.</returns>
    public IReadOnlyList<ReasoningModelDiagnostic> GetRecent()
    {
        lock (_gate)
        {
            return new ReadOnlyCollection<ReasoningModelDiagnostic>(_recent.ToArray());
        }
    }

    /// <summary>Gets one representative record per distinct adjustment, with how often it occurred.</summary>
    /// <returns>
    /// A dictionary keyed by <see cref="ReasoningModelDiagnostic.Key"/> mapping to the first record seen for that
    /// key and its occurrence count. Bounded by the number of distinct settings, not by the number of requests.
    /// </returns>
    public IReadOnlyDictionary<string, ReasoningModelDiagnosticSummary> GetSummary()
    {
        lock (_gate)
        {
            var summary = new Dictionary<string, ReasoningModelDiagnosticSummary>(_distinct.Count, StringComparer.Ordinal);
            foreach (KeyValuePair<string, ReasoningModelDiagnostic> pair in _distinct)
            {
                summary[pair.Key] = new ReasoningModelDiagnosticSummary(pair.Value, _counts[pair.Key]);
            }

            return new ReadOnlyDictionary<string, ReasoningModelDiagnosticSummary>(summary);
        }
    }

    /// <summary>Discards every retained record and resets the counts.</summary>
    public void Clear()
    {
        lock (_gate)
        {
            _recent.Clear();
            _distinct.Clear();
            _counts.Clear();
            _total = 0L;
        }
    }
}
