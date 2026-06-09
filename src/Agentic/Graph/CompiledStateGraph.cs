using System.Runtime.CompilerServices;

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
    private readonly string _entryPoint;

    internal CompiledStateGraph(
        IReadOnlyDictionary<string, Func<TState, CancellationToken, Task<TState>>> nodes,
        IReadOnlyDictionary<string, string> edges,
        IReadOnlyDictionary<string, Func<TState, string>> conditionalEdges,
        string entryPoint)
    {
        _nodes = nodes;
        _edges = edges;
        _conditionalEdges = conditionalEdges;
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

            state = await _nodes[current](state, cancellationToken).ConfigureAwait(false);
            current = NextNode(current, state);
        }

        return state;
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
