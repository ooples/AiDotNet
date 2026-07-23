namespace AiDotNet.Agentic.Graph.Checkpointing;

/// <summary>
/// An in-memory <see cref="IGraphCheckpointer{TState}"/> that keeps each thread's checkpoint history in a
/// dictionary. Suitable for tests, single-process runs, and as the default when no durable store is wired.
/// </summary>
/// <typeparam name="TState">The graph's state type.</typeparam>
/// <remarks>
/// <para>
/// Thread-safe via a per-instance lock. Because it stores the state object by reference, callers that use
/// a mutable reference type for <typeparamref name="TState"/> should treat the state as immutable per step
/// (return a new/copied instance from nodes) to get true snapshots; durable checkpointers serialize and so
/// snapshot inherently.
/// </para>
/// <para><b>For Beginners:</b> Keeps your save-games in memory. Great for tests and short-lived runs; for
/// durability across process restarts, use a database-backed checkpointer (coming in later slices).
/// </para>
/// </remarks>
public sealed class InMemoryGraphCheckpointer<TState> : IGraphCheckpointer<TState>
{
    private readonly object _gate = new();
    private readonly Dictionary<string, List<GraphCheckpoint<TState>>> _byThread = new(StringComparer.Ordinal);

    /// <inheritdoc/>
    public Task SaveAsync(GraphCheckpoint<TState> checkpoint, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(checkpoint);
        lock (_gate)
        {
            if (!_byThread.TryGetValue(checkpoint.ThreadId, out var list))
            {
                list = new List<GraphCheckpoint<TState>>();
                _byThread[checkpoint.ThreadId] = list;
            }

            list.Add(checkpoint);
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task<GraphCheckpoint<TState>?> GetLatestAsync(string threadId, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(threadId);
        lock (_gate)
        {
            if (_byThread.TryGetValue(threadId, out var list) && list.Count > 0)
            {
                return Task.FromResult<GraphCheckpoint<TState>?>(list[list.Count - 1]);
            }
        }

        return Task.FromResult<GraphCheckpoint<TState>?>(null);
    }

    /// <inheritdoc/>
    public Task<GraphCheckpoint<TState>?> GetAsync(string threadId, string checkpointId, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(threadId);
        Guard.NotNull(checkpointId);
        lock (_gate)
        {
            if (_byThread.TryGetValue(threadId, out var list))
            {
                // Return the NEWEST checkpoint with this id (history is append-ordered), matching the durable
                // backends which order by descending sequence — so re-saving an id resolves to the latest.
                var match = list.LastOrDefault(cp => cp.CheckpointId == checkpointId);
                return Task.FromResult<GraphCheckpoint<TState>?>(match);
            }
        }

        return Task.FromResult<GraphCheckpoint<TState>?>(null);
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<GraphCheckpoint<TState>>> GetHistoryAsync(string threadId, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(threadId);
        lock (_gate)
        {
            IReadOnlyList<GraphCheckpoint<TState>> history = _byThread.TryGetValue(threadId, out var list)
                ? list.ToList()
                : new List<GraphCheckpoint<TState>>();
            return Task.FromResult(history);
        }
    }
}
