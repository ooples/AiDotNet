using System.Collections.ObjectModel;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.Provenance;

/// <summary>A bounded in-memory <see cref="IProposalProvenanceSink"/> for tests and short-lived runs.</summary>
/// <remarks>
/// <para>
/// Keeps the most recent records in insertion order and discards the oldest once the capacity is reached, so it
/// cannot grow without limit on a long run. Nothing is written to disk, which makes it the right sink for a test
/// that wants to assert what the operator recorded, and for an interactive session that only needs to look at the
/// last few exchanges.
/// </para>
/// <para>
/// Reads and writes are serialized, so one instance may be shared by concurrent workers, and
/// <see cref="GetRecords"/> returns a snapshot that is safe to enumerate while a run continues.
/// </para>
/// <para><b>For Beginners:</b> The same notebook as the file-writing sink, but held in memory and capped at a fixed
/// number of pages. Useful when you just want to look at what happened, and do not want files left behind.</para>
/// </remarks>
public sealed class InMemoryProposalProvenanceSink : IProposalProvenanceSink
{
    /// <summary>The number of records retained when no capacity is supplied.</summary>
    public const int DefaultCapacity = 1_024;

    private readonly object _gate = new();
    private readonly Queue<ProposalProvenanceRecord> _records = new();
    private readonly int _capacity;
    private long _total;

    /// <summary>Initializes an in-memory sink.</summary>
    /// <param name="capacity">How many records to retain; defaults to <see cref="DefaultCapacity"/>.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="capacity"/> is not positive.</exception>
    public InMemoryProposalProvenanceSink(int capacity = DefaultCapacity)
    {
        Guard.Positive(capacity);
        _capacity = capacity;
    }

    /// <summary>Gets how many records have been offered since construction, including discarded ones.</summary>
    public long TotalRecorded
    {
        get
        {
            lock (_gate) return _total;
        }
    }

    /// <inheritdoc/>
    public Task RecordAsync(ProposalProvenanceRecord record, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(record);
        cancellationToken.ThrowIfCancellationRequested();
        lock (_gate)
        {
            _total++;
            _records.Enqueue(record);
            while (_records.Count > _capacity) _records.Dequeue();
        }

        return Task.CompletedTask;
    }

    /// <summary>Gets the retained records, oldest first.</summary>
    /// <returns>A snapshot holding up to the configured capacity of records.</returns>
    public IReadOnlyList<ProposalProvenanceRecord> GetRecords()
    {
        lock (_gate)
        {
            return new ReadOnlyCollection<ProposalProvenanceRecord>(_records.ToArray());
        }
    }

    /// <summary>Discards every retained record and resets the total.</summary>
    public void Clear()
    {
        lock (_gate)
        {
            _records.Clear();
            _total = 0L;
        }
    }
}
