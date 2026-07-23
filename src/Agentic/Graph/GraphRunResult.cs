namespace AiDotNet.Agentic.Graph;

/// <summary>
/// The outcome of a human-in-the-loop graph run: either the run completed, or it paused before an
/// interrupt node awaiting input. Carries the state either way.
/// </summary>
/// <typeparam name="TState">The graph's state type.</typeparam>
/// <remarks>
/// <para>
/// When a graph has interrupt points (see <see cref="StateGraph{TState}.AddInterruptBefore"/>), a run
/// stops just before such a node and returns an interrupted result with <see cref="InterruptedBefore"/>
/// set to that node. Call the run again on the same thread to resume past the pause (optionally editing
/// the state with the human's input first).
/// </para>
/// <para><b>For Beginners:</b> Tells you whether the graph finished or stopped to ask a human. If it
/// stopped, <see cref="InterruptedBefore"/> says which step it's waiting to run, and <see cref="State"/>
/// is the data so far so you can review/edit it before continuing.
/// </para>
/// </remarks>
public sealed class GraphRunResult<TState>
{
    private GraphRunResult(bool isComplete, TState state, string? interruptedBefore)
    {
        IsComplete = isComplete;
        State = state;
        InterruptedBefore = interruptedBefore;
    }

    /// <summary>Gets a value indicating whether the run reached the end node.</summary>
    public bool IsComplete { get; }

    /// <summary>Gets a value indicating whether the run paused at an interrupt point.</summary>
    public bool IsInterrupted => InterruptedBefore is not null;

    /// <summary>Gets the state (final when complete, or as of the pause when interrupted).</summary>
    public TState State { get; }

    /// <summary>Gets the node the run paused before, or <c>null</c> when the run completed.</summary>
    public string? InterruptedBefore { get; }

    /// <summary>Creates a completed result.</summary>
    /// <param name="state">The final state.</param>
    /// <returns>A completed <see cref="GraphRunResult{TState}"/>.</returns>
    public static GraphRunResult<TState> Complete(TState state) => new(true, state, null);

    /// <summary>Creates an interrupted result.</summary>
    /// <param name="state">The state as of the pause.</param>
    /// <param name="node">The node the run paused before.</param>
    /// <returns>An interrupted <see cref="GraphRunResult{TState}"/>.</returns>
    public static GraphRunResult<TState> Interrupted(TState state, string node)
    {
        Guard.NotNullOrWhiteSpace(node);
        return new GraphRunResult<TState>(false, state, node);
    }
}
