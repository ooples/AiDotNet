namespace AiDotNet.Agentic.Graph.Checkpointing;

/// <summary>
/// Persists and retrieves <see cref="GraphCheckpoint{TState}"/>s, enabling durable resume and time-travel
/// for graph runs. Implementations back onto memory, SQLite, Postgres, Redis, etc.
/// </summary>
/// <typeparam name="TState">The graph's state type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The save-game store. The graph asks it to save a checkpoint after each
/// step, to fetch the latest checkpoint when resuming, and to list the full history when rewinding.
/// </para>
/// </remarks>
public interface IGraphCheckpointer<TState>
{
    /// <summary>
    /// Saves (appends) a checkpoint.
    /// </summary>
    /// <param name="checkpoint">The checkpoint to persist.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    Task SaveAsync(GraphCheckpoint<TState> checkpoint, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the most recent checkpoint for a thread, or <c>null</c> if the thread has none.
    /// </summary>
    /// <param name="threadId">The run/thread id.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    /// <returns>The latest checkpoint, or <c>null</c>.</returns>
    Task<GraphCheckpoint<TState>?> GetLatestAsync(string threadId, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets a specific checkpoint by id (used for time-travel / replay from a past point).
    /// </summary>
    /// <param name="threadId">The run/thread id.</param>
    /// <param name="checkpointId">The checkpoint id.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    /// <returns>The checkpoint, or <c>null</c> if not found.</returns>
    Task<GraphCheckpoint<TState>?> GetAsync(string threadId, string checkpointId, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the full <see cref="GraphCheckpoint{TState}"/> history for <paramref name="threadId"/> in
    /// append (chronological) order — i.e. the order checkpoints were saved, by a persistent monotonic
    /// sequence (e.g. the SQLite backend orders by <c>Seq ASC</c>), NOT by logical step number.
    /// </summary>
    /// <remarks>
    /// Replay/time-travel relies on this chronological ordering: logical step values can repeat or branch
    /// (loops, re-runs from an earlier checkpoint), so ordering by step would diverge across backends.
    /// Implementations MUST return checkpoints oldest-first by save order.
    /// </remarks>
    /// <param name="threadId">The run/thread id.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    /// <returns>The chronologically-ordered history (possibly empty).</returns>
    Task<IReadOnlyList<GraphCheckpoint<TState>>> GetHistoryAsync(string threadId, CancellationToken cancellationToken = default);
}
