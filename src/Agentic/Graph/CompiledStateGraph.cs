using System.Runtime.CompilerServices;
using AiDotNet.Agentic.Graph.Checkpointing;

namespace AiDotNet.Agentic.Graph;

/// <summary>
/// An executable, validated state graph produced by <see cref="StateGraph{TState}.Compile"/>. Runs nodes
/// according to the graph's edges, threading the state from one node to the next until it reaches the end.
/// </summary>
/// <typeparam name="TState">The graph's state type (threaded through every node).</typeparam>
/// <remarks>
/// <para>
/// Execution starts at the entry node. After a node runs, the next node is chosen by: its conditional
/// router (if any), else its fixed edge, else the run ends. Cycles are allowed and bounded by
/// <see cref="GraphRunOptions.MaxSteps"/>. Use <see cref="InvokeAsync"/> for the final state or
/// <see cref="StreamAsync"/> to observe each step.
/// </para>
/// <para><b>For Beginners:</b> This is the "compiled program" form of your graph. You give it a starting
/// state and it walks the nodes you wired up, handing each node the latest state, until it reaches the
/// end — then returns the final state.
/// </para>
/// </remarks>
public sealed class CompiledStateGraph<TState>
{
    private readonly IReadOnlyDictionary<string, Func<TState, CancellationToken, Task<TState>>> _nodes;
    private readonly IReadOnlyDictionary<string, string> _edges;
    private readonly IReadOnlyDictionary<string, Func<TState, string>> _conditionalEdges;
    private readonly HashSet<string> _interruptBefore;
    private readonly string _entryPoint;

    internal CompiledStateGraph(
        IReadOnlyDictionary<string, Func<TState, CancellationToken, Task<TState>>> nodes,
        IReadOnlyDictionary<string, string> edges,
        IReadOnlyDictionary<string, Func<TState, string>> conditionalEdges,
        HashSet<string> interruptBefore,
        string entryPoint)
    {
        _nodes = nodes;
        _edges = edges;
        _conditionalEdges = conditionalEdges;
        _interruptBefore = interruptBefore;
        _entryPoint = entryPoint;
    }

    /// <summary>
    /// Runs the graph from the supplied initial state and returns the final state.
    /// </summary>
    /// <param name="initialState">The starting state.</param>
    /// <param name="options">Run options (step budget). <c>null</c> uses defaults.</param>
    /// <param name="cancellationToken">Token used to cancel the run.</param>
    /// <returns>The final state when flow reaches the end node.</returns>
    /// <exception cref="GraphRecursionException">Thrown when the step budget is exceeded.</exception>
    public async Task<TState> InvokeAsync(
        TState initialState,
        GraphRunOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var result = await RunCoreAsync(
            initialState, _entryPoint, startStep: 0, ResolveMaxSteps(options),
            checkpointer: null, threadId: null, honorInterrupts: false, honorFirstInterrupt: false, cancellationToken).ConfigureAwait(false);
        return result.State;
    }

    /// <summary>
    /// Runs the graph with durable checkpointing under a thread id, resuming automatically from the latest
    /// saved checkpoint if one exists for that thread (otherwise starting fresh from <paramref name="initialState"/>).
    /// </summary>
    /// <param name="initialState">The starting state (used only when the thread has no prior checkpoint).</param>
    /// <param name="checkpointer">The checkpoint store.</param>
    /// <param name="threadId">The run/thread id under which checkpoints are saved and resumed.</param>
    /// <param name="options">Run options (step budget). <c>null</c> uses defaults. The budget is cumulative across resumes.</param>
    /// <param name="cancellationToken">Token used to cancel the run.</param>
    /// <returns>The final state.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="checkpointer"/> or <paramref name="threadId"/> is <c>null</c>.</exception>
    /// <exception cref="GraphRecursionException">Thrown when the step budget is exceeded.</exception>
    public async Task<TState> InvokeAsync(
        TState initialState,
        IGraphCheckpointer<TState> checkpointer,
        string threadId,
        GraphRunOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(checkpointer);
        Guard.NotNullOrWhiteSpace(threadId);
        var maxSteps = ResolveMaxSteps(options);

        var latest = await checkpointer.GetLatestAsync(threadId, cancellationToken).ConfigureAwait(false);
        string startNode;
        TState state;
        int startStep;
        if (latest is not null)
        {
            startNode = latest.NextNode;
            state = latest.State;
            startStep = latest.Step;
        }
        else
        {
            startNode = _entryPoint;
            state = initialState;
            startStep = 0;
            await checkpointer.SaveAsync(
                new GraphCheckpoint<TState>(threadId, $"{threadId}-0", 0, _entryPoint, initialState), cancellationToken).ConfigureAwait(false);
        }

        if (startNode == GraphSpecialNodes.End)
        {
            return state; // thread already complete
        }

        var result = await RunCoreAsync(
            state, startNode, startStep, maxSteps, checkpointer, threadId,
            honorInterrupts: false, honorFirstInterrupt: false, cancellationToken).ConfigureAwait(false);
        return result.State;
    }

    /// <summary>
    /// Resumes (replays) a thread from a specific past checkpoint — the basis for time-travel — continuing
    /// execution from that checkpoint's next node and state, and saving new checkpoints as it goes.
    /// </summary>
    /// <param name="checkpointer">The checkpoint store.</param>
    /// <param name="threadId">The run/thread id.</param>
    /// <param name="checkpointId">The id of the checkpoint to resume from.</param>
    /// <param name="options">Run options (step budget). <c>null</c> uses defaults.</param>
    /// <param name="cancellationToken">Token used to cancel the run.</param>
    /// <returns>The final state.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="checkpointer"/>, <paramref name="threadId"/>, or <paramref name="checkpointId"/> is <c>null</c>.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the checkpoint does not exist.</exception>
    public async Task<TState> ResumeFromAsync(
        IGraphCheckpointer<TState> checkpointer,
        string threadId,
        string checkpointId,
        GraphRunOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(checkpointer);
        Guard.NotNullOrWhiteSpace(threadId);
        Guard.NotNullOrWhiteSpace(checkpointId);

        var checkpoint = await checkpointer.GetAsync(threadId, checkpointId, cancellationToken).ConfigureAwait(false);
        if (checkpoint is null)
        {
            throw new InvalidOperationException($"Checkpoint '{checkpointId}' was not found for thread '{threadId}'.");
        }

        if (checkpoint.NextNode == GraphSpecialNodes.End)
        {
            return checkpoint.State;
        }

        var result = await RunCoreAsync(
            checkpoint.State, checkpoint.NextNode, checkpoint.Step, ResolveMaxSteps(options), checkpointer, threadId,
            honorInterrupts: false, honorFirstInterrupt: false, cancellationToken).ConfigureAwait(false);
        return result.State;
    }

    /// <summary>
    /// Runs the graph with human-in-the-loop interrupts under a thread id. On the first call it runs from
    /// the entry node and pauses just before the first interrupt node (returning an interrupted result);
    /// calling it again on the same thread resumes past that pause and runs until the next interrupt or
    /// completion. Optionally applies the human's edit to the state when resuming.
    /// </summary>
    /// <param name="initialState">The starting state (used only when the thread has no prior checkpoint).</param>
    /// <param name="checkpointer">The checkpoint store.</param>
    /// <param name="threadId">The run/thread id.</param>
    /// <param name="applyOnResume">When resuming, an optional transform applied to the saved state (the human's input). Ignored on a fresh run.</param>
    /// <param name="options">Run options (step budget). <c>null</c> uses defaults.</param>
    /// <param name="cancellationToken">Token used to cancel the run.</param>
    /// <returns>A <see cref="GraphRunResult{TState}"/> indicating completion or an interrupt point.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="checkpointer"/> or <paramref name="threadId"/> is <c>null</c>.</exception>
    public async Task<GraphRunResult<TState>> RunAsync(
        TState initialState,
        IGraphCheckpointer<TState> checkpointer,
        string threadId,
        Func<TState, TState>? applyOnResume = null,
        GraphRunOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(checkpointer);
        Guard.NotNullOrWhiteSpace(threadId);
        var maxSteps = ResolveMaxSteps(options);

        var latest = await checkpointer.GetLatestAsync(threadId, cancellationToken).ConfigureAwait(false);
        string startNode;
        TState state;
        int startStep;
        bool honorFirstInterrupt;
        if (latest is not null)
        {
            // Resuming: the caller chose to continue, so do not re-pause at the node we stopped before.
            startNode = latest.NextNode;
            state = applyOnResume is not null ? applyOnResume(latest.State) : latest.State;
            startStep = latest.Step;
            honorFirstInterrupt = false;
        }
        else
        {
            startNode = _entryPoint;
            state = initialState;
            startStep = 0;
            honorFirstInterrupt = true;
            await checkpointer.SaveAsync(
                new GraphCheckpoint<TState>(threadId, $"{threadId}-0", 0, _entryPoint, initialState), cancellationToken).ConfigureAwait(false);
        }

        if (startNode == GraphSpecialNodes.End)
        {
            return GraphRunResult<TState>.Complete(state);
        }

        return await RunCoreAsync(
            state, startNode, startStep, maxSteps, checkpointer, threadId,
            honorInterrupts: true, honorFirstInterrupt: honorFirstInterrupt, cancellationToken).ConfigureAwait(false);
    }

    private async Task<GraphRunResult<TState>> RunCoreAsync(
        TState state,
        string startNode,
        int startStep,
        int maxSteps,
        IGraphCheckpointer<TState>? checkpointer,
        string? threadId,
        bool honorInterrupts,
        bool honorFirstInterrupt,
        CancellationToken cancellationToken)
    {
        var current = startNode;
        var step = startStep;
        var firstNode = true;

        while (current != GraphSpecialNodes.End)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var considerInterrupt = honorInterrupts && (honorFirstInterrupt || !firstNode);
            if (considerInterrupt && _interruptBefore.Contains(current))
            {
                // Pause before this node. The latest checkpoint already records NextNode == current,
                // so a subsequent resume picks up here.
                return GraphRunResult<TState>.Interrupted(state, current);
            }

            firstNode = false;

            if (++step > maxSteps)
            {
                throw new GraphRecursionException(maxSteps);
            }

            state = await _nodes[current](state, cancellationToken).ConfigureAwait(false);
            current = NextNode(current, state);

            if (checkpointer is not null && threadId is not null)
            {
                await checkpointer.SaveAsync(
                    new GraphCheckpoint<TState>(threadId, $"{threadId}-{step}", step, current, state), cancellationToken).ConfigureAwait(false);
            }
        }

        return GraphRunResult<TState>.Complete(state);
    }

    /// <summary>
    /// Runs the graph and yields an update after each node executes, ending when flow reaches the end node.
    /// </summary>
    /// <param name="initialState">The starting state.</param>
    /// <param name="options">Run options (step budget). <c>null</c> uses defaults.</param>
    /// <param name="cancellationToken">Token used to cancel the run.</param>
    /// <returns>A stream of <see cref="GraphStepUpdate{TState}"/>; the final update carries the final state.</returns>
    /// <exception cref="GraphRecursionException">Thrown when the step budget is exceeded.</exception>
    public async IAsyncEnumerable<GraphStepUpdate<TState>> StreamAsync(
        TState initialState,
        GraphRunOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var maxSteps = ResolveMaxSteps(options);
        var state = initialState;
        var current = _entryPoint;
        var steps = 0;

        while (current != GraphSpecialNodes.End)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (++steps > maxSteps)
            {
                throw new GraphRecursionException(maxSteps);
            }

            var executed = current;
            state = await _nodes[current](state, cancellationToken).ConfigureAwait(false);
            yield return new GraphStepUpdate<TState>(executed, state);
            current = NextNode(current, state);
        }
    }

    /// <summary>
    /// Deterministically replays a recorded run from its checkpoint history WITHOUT executing any nodes,
    /// yielding the same (node, state) sequence the original run produced.
    /// </summary>
    /// <param name="checkpointer">The checkpoint store holding the recorded run.</param>
    /// <param name="threadId">The recorded run/thread id.</param>
    /// <param name="cancellationToken">Token used to cancel the replay.</param>
    /// <returns>The recorded per-node updates, in order.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Re-watch a past run exactly as it happened, reading from the saved
    /// checkpoints instead of re-running the (possibly slow or non-deterministic) nodes. Great for
    /// debugging and audit.
    /// </para>
    /// </remarks>
    public async IAsyncEnumerable<GraphStepUpdate<TState>> ReplayAsync(
        IGraphCheckpointer<TState> checkpointer,
        string threadId,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Guard.NotNull(checkpointer);
        Guard.NotNullOrWhiteSpace(threadId);

        var history = await checkpointer.GetHistoryAsync(threadId, cancellationToken).ConfigureAwait(false);
        for (var k = 1; k < history.Count; k++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            // The node that ran to produce history[k] is the one recorded as "next" by history[k-1].
            var executedNode = history[k - 1].NextNode;
            yield return new GraphStepUpdate<TState>(executedNode, history[k].State);
        }
    }

    /// <summary>
    /// Returns the final recorded state for a thread (read from checkpoints, no execution).
    /// </summary>
    /// <param name="checkpointer">The checkpoint store.</param>
    /// <param name="threadId">The run/thread id.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    /// <returns>The latest recorded state.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the thread has no checkpoints.</exception>
    public async Task<TState> GetRecordedStateAsync(
        IGraphCheckpointer<TState> checkpointer,
        string threadId,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(checkpointer);
        Guard.NotNullOrWhiteSpace(threadId);

        var latest = await checkpointer.GetLatestAsync(threadId, cancellationToken).ConfigureAwait(false);
        if (latest is null)
        {
            throw new InvalidOperationException($"No checkpoints found for thread '{threadId}'.");
        }

        return latest.State;
    }

    private static int ResolveMaxSteps(GraphRunOptions? options)
    {
        var max = options?.MaxSteps ?? new GraphRunOptions().MaxSteps;
        return max > 0 ? max : new GraphRunOptions().MaxSteps;
    }

    private string NextNode(string current, TState state)
    {
        if (_conditionalEdges.TryGetValue(current, out var router))
        {
            var next = router(state);
            if (string.IsNullOrEmpty(next))
            {
                throw new InvalidOperationException(
                    $"The conditional router for node '{current}' returned a null or empty target.");
            }

            if (next != GraphSpecialNodes.End && !_nodes.ContainsKey(next))
            {
                throw new InvalidOperationException(
                    $"The conditional router for node '{current}' returned unknown target '{next}'.");
            }

            return next;
        }

        return _edges.TryGetValue(current, out var target) ? target : GraphSpecialNodes.End;
    }
}
