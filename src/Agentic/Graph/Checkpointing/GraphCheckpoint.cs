namespace AiDotNet.Agentic.Graph.Checkpointing;

/// <summary>
/// An immutable snapshot of a graph run at a point in time: which node runs next and the state as of
/// that point. Saved after each step so a run can be resumed, inspected, or replayed.
/// </summary>
/// <typeparam name="TState">The graph's state type.</typeparam>
/// <remarks>
/// <para>
/// A checkpoint records the <see cref="NextNode"/> (where execution will continue) plus the
/// <see cref="State"/> at that boundary, tagged with a <see cref="ThreadId"/> (the run/conversation) and a
/// monotonically increasing <see cref="Step"/>. The sequence of checkpoints for a thread is its history —
/// the basis for durable resume and time-travel.
/// </para>
/// <para><b>For Beginners:</b> Like a save-game. It remembers exactly where the graph was about to go
/// next and what the data looked like, so you can stop and pick up later (or rewind to an earlier save).
/// </para>
/// </remarks>
public sealed class GraphCheckpoint<TState>
{
    /// <summary>
    /// Initializes a new checkpoint.
    /// </summary>
    /// <param name="threadId">The run/thread this checkpoint belongs to.</param>
    /// <param name="checkpointId">A unique id for this checkpoint within the thread.</param>
    /// <param name="step">The zero-based step index (monotonically increasing within a thread).</param>
    /// <param name="nextNode">The node that will execute next (or <see cref="GraphSpecialNodes.End"/> when complete).</param>
    /// <param name="state">The state at this boundary.</param>
    /// <exception cref="ArgumentNullException">Thrown when a required string is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when a required string is empty/whitespace.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="step"/> is negative.</exception>
    public GraphCheckpoint(string threadId, string checkpointId, int step, string nextNode, TState state)
    {
        Guard.NotNullOrWhiteSpace(threadId);
        Guard.NotNullOrWhiteSpace(checkpointId);
        Guard.NotNullOrWhiteSpace(nextNode);
        Guard.NonNegative(step);
        ThreadId = threadId;
        CheckpointId = checkpointId;
        Step = step;
        NextNode = nextNode;
        State = state;
    }

    /// <summary>Gets the run/thread this checkpoint belongs to.</summary>
    public string ThreadId { get; }

    /// <summary>Gets the unique id of this checkpoint within the thread.</summary>
    public string CheckpointId { get; }

    /// <summary>Gets the zero-based, monotonically increasing step index.</summary>
    public int Step { get; }

    /// <summary>Gets the node that will execute next (or <see cref="GraphSpecialNodes.End"/> when the run is complete).</summary>
    public string NextNode { get; }

    /// <summary>Gets the state captured at this boundary.</summary>
    public TState State { get; }

    /// <summary>Gets a value indicating whether this checkpoint represents a completed run.</summary>
    public bool IsComplete => NextNode == GraphSpecialNodes.End;
}
