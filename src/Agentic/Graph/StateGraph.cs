namespace AiDotNet.Agentic.Graph;

/// <summary>
/// A builder for a typed state graph: register nodes (state transformers), wire them with fixed or
/// conditional edges (cycles allowed), set an entry point, then <see cref="Compile"/> into an executable
/// <see cref="CompiledStateGraph{TState}"/>.
/// </summary>
/// <typeparam name="TState">
/// The state type threaded through the graph. Each node receives the current state and returns the next
/// state (it may mutate and return the same instance, or return a new one).
/// </typeparam>
/// <remarks>
/// <para>
/// This is the AiDotNet counterpart to LangGraph's <c>StateGraph</c>, but fully typed: there are no
/// stringly-typed state dictionaries — <typeparamref name="TState"/> is your own type, checked by the
/// compiler. Routing is explicit: a node has at most one fixed edge or one conditional router; reaching
/// <see cref="End"/> finishes the run.
/// </para>
/// <para><b>For Beginners:</b> Think of building a flowchart. <see cref="AddNode"/> adds a box that does
/// some work on your data; <see cref="AddEdge"/> draws an arrow to the next box; <see cref="AddConditionalEdges"/>
/// draws arrows that depend on the data; <see cref="SetEntryPoint"/> marks where to start. Arrows can
/// loop back to create cycles. <see cref="Compile"/> turns the flowchart into something you can run.
/// </para>
/// <para><b>Example:</b>
/// <code>
/// var graph = new StateGraph&lt;MyState&gt;()
///     .AddNode("plan", (s, ct) =&gt; Task.FromResult(s.WithPlan()))
///     .AddNode("act", (s, ct) =&gt; Task.FromResult(s.Act()))
///     .AddConditionalEdges("act", s =&gt; s.IsDone ? StateGraph&lt;MyState&gt;.End : "act")
///     .AddEdge("plan", "act")
///     .SetEntryPoint("plan")
///     .Compile();
/// var result = await graph.InvokeAsync(new MyState());
/// </code>
/// </para>
/// </remarks>
public sealed class StateGraph<TState>
{
    /// <summary>
    /// The terminal node name; route here to finish the run. Alias of <see cref="GraphSpecialNodes.End"/>.
    /// </summary>
    public const string End = GraphSpecialNodes.End;

    private readonly Dictionary<string, Func<TState, CancellationToken, Task<TState>>> _nodes = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _edges = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Func<TState, string>> _conditionalEdges = new(StringComparer.Ordinal);
    private readonly HashSet<string> _interruptBefore = new(StringComparer.Ordinal);
    private string? _entryPoint;

    /// <summary>
    /// Adds an asynchronous node that transforms the state.
    /// </summary>
    /// <param name="name">The unique node name (cannot be empty or the reserved end name).</param>
    /// <param name="node">The node body: receives the current state and returns the next state.</param>
    /// <returns>This builder, for chaining.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="name"/> or <paramref name="node"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when the name is empty/whitespace, reserved, or already used.</exception>
    public StateGraph<TState> AddNode(string name, Func<TState, CancellationToken, Task<TState>> node)
    {
        Guard.NotNullOrWhiteSpace(name);
        Guard.NotNull(node);
        if (name == End)
        {
            throw new ArgumentException($"'{End}' is a reserved node name.", nameof(name));
        }

        if (_nodes.ContainsKey(name))
        {
            throw new ArgumentException($"A node named '{name}' has already been added.", nameof(name));
        }

        _nodes[name] = node;
        return this;
    }

    /// <summary>
    /// Adds a synchronous node that transforms the state.
    /// </summary>
    /// <param name="name">The unique node name.</param>
    /// <param name="node">The node body: receives the current state and returns the next state.</param>
    /// <returns>This builder, for chaining.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="name"/> or <paramref name="node"/> is <c>null</c>.</exception>
    public StateGraph<TState> AddNode(string name, Func<TState, TState> node)
    {
        Guard.NotNull(node);
        return AddNode(name, (state, _) => Task.FromResult(node(state)));
    }

    /// <summary>
    /// Adds a fixed edge: after <paramref name="from"/> runs, flow always moves to <paramref name="to"/>.
    /// </summary>
    /// <param name="from">The source node name.</param>
    /// <param name="to">The destination node name, or <see cref="End"/>.</param>
    /// <returns>This builder, for chaining.</returns>
    /// <exception cref="ArgumentException">Thrown when names are empty or <paramref name="from"/> already has a fixed edge.</exception>
    public StateGraph<TState> AddEdge(string from, string to)
    {
        Guard.NotNullOrWhiteSpace(from);
        Guard.NotNullOrWhiteSpace(to);
        if (_edges.ContainsKey(from))
        {
            throw new ArgumentException($"Node '{from}' already has a fixed edge.", nameof(from));
        }

        _edges[from] = to;
        return this;
    }

    /// <summary>
    /// Adds conditional edges: after <paramref name="from"/> runs, the router inspects the state and
    /// returns the next node name (or <see cref="End"/>).
    /// </summary>
    /// <param name="from">The source node name.</param>
    /// <param name="router">Selects the next node from the current state.</param>
    /// <returns>This builder, for chaining.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="router"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="from"/> is empty or already has conditional edges.</exception>
    public StateGraph<TState> AddConditionalEdges(string from, Func<TState, string> router)
    {
        Guard.NotNullOrWhiteSpace(from);
        Guard.NotNull(router);
        if (_conditionalEdges.ContainsKey(from))
        {
            throw new ArgumentException($"Node '{from}' already has conditional edges.", nameof(from));
        }

        _conditionalEdges[from] = router;
        return this;
    }

    /// <summary>
    /// Adds a node that runs an entire compiled graph as a single step (a subgraph), threading the same
    /// state in and out. Enables composition: build small graphs and embed them in larger ones.
    /// </summary>
    /// <param name="name">The node name for the subgraph.</param>
    /// <param name="subgraph">The compiled graph to run when this node executes.</param>
    /// <returns>This builder, for chaining.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="subgraph"/> is <c>null</c>.</exception>
    public StateGraph<TState> AddSubgraph(string name, CompiledStateGraph<TState> subgraph)
    {
        Guard.NotNull(subgraph);
        return AddNode(name, (state, ct) => subgraph.InvokeAsync(state, null, ct));
    }

    /// <summary>
    /// Adds reward-gated routing after a node: a scoring function rates the state, and flow goes to
    /// <paramref name="ifMeetsThreshold"/> when the score is at least <paramref name="threshold"/>, else to
    /// <paramref name="ifBelowThreshold"/>. A differentiator for verifier/critic-driven control flow.
    /// </summary>
    /// <param name="from">The source node name.</param>
    /// <param name="reward">Scores the current state (e.g., a critic/reward model).</param>
    /// <param name="threshold">The minimum score required to take the "meets" branch.</param>
    /// <param name="ifMeetsThreshold">The next node when <c>reward(state) &gt;= threshold</c> (may be <see cref="End"/>).</param>
    /// <param name="ifBelowThreshold">The next node otherwise (may be <see cref="End"/>).</param>
    /// <returns>This builder, for chaining.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="reward"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when a node name is empty/whitespace.</exception>
    public StateGraph<TState> AddRewardGatedEdges(
        string from,
        Func<TState, double> reward,
        double threshold,
        string ifMeetsThreshold,
        string ifBelowThreshold)
    {
        Guard.NotNull(reward);
        Guard.NotNullOrWhiteSpace(ifMeetsThreshold);
        Guard.NotNullOrWhiteSpace(ifBelowThreshold);
        return AddConditionalEdges(from, state => reward(state) >= threshold ? ifMeetsThreshold : ifBelowThreshold);
    }

    /// <summary>
    /// Adds a dynamic fan-out (map-reduce) node: it derives a set of items from the current state, runs a
    /// branch over each item in parallel, then reduces the branch results back into the state. This is the
    /// typed equivalent of LangGraph's <c>Send</c>/map-reduce.
    /// </summary>
    /// <typeparam name="TItem">The per-branch input item type.</typeparam>
    /// <typeparam name="TResult">The per-branch result type.</typeparam>
    /// <param name="name">The node name.</param>
    /// <param name="map">Derives the items to fan out over from the current state.</param>
    /// <param name="branch">The work run for each item (in parallel).</param>
    /// <param name="reduce">Merges the ordered branch results back into the state.</param>
    /// <param name="maxDegreeOfParallelism">Optional cap on concurrent branches; <c>null</c> runs all at once.</param>
    /// <returns>This builder, for chaining.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="map"/>, <paramref name="branch"/>, or <paramref name="reduce"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="maxDegreeOfParallelism"/> is less than 1.</exception>
    public StateGraph<TState> AddFanOutNode<TItem, TResult>(
        string name,
        Func<TState, IEnumerable<TItem>> map,
        Func<TItem, CancellationToken, Task<TResult>> branch,
        Func<TState, IReadOnlyList<TResult>, TState> reduce,
        int? maxDegreeOfParallelism = null)
    {
        Guard.NotNull(map);
        Guard.NotNull(branch);
        Guard.NotNull(reduce);
        if (maxDegreeOfParallelism is { } dop && dop < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxDegreeOfParallelism), "Max degree of parallelism must be at least 1.");
        }

        return AddNode(name, async (state, ct) =>
        {
            var items = (map(state) ?? Enumerable.Empty<TItem>()).ToList();
            IReadOnlyList<TResult> results = await RunBranchesAsync(items, branch, maxDegreeOfParallelism, ct).ConfigureAwait(false);
            return reduce(state, results);
        });
    }

    private static async Task<TResult[]> RunBranchesAsync<TItem, TResult>(
        IReadOnlyList<TItem> items,
        Func<TItem, CancellationToken, Task<TResult>> branch,
        int? maxDegreeOfParallelism,
        CancellationToken cancellationToken)
    {
        if (maxDegreeOfParallelism is not { } dop)
        {
            return await Task.WhenAll(items.Select(item => branch(item, cancellationToken))).ConfigureAwait(false);
        }

        using var throttle = new SemaphoreSlim(dop);
        async Task<TResult> RunThrottled(TItem item)
        {
            await throttle.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                return await branch(item, cancellationToken).ConfigureAwait(false);
            }
            finally
            {
                throttle.Release();
            }
        }

        return await Task.WhenAll(items.Select(RunThrottled)).ConfigureAwait(false);
    }

    /// <summary>
    /// Marks a node as a human-in-the-loop interrupt point: a run pauses just before this node so a human
    /// can review (and optionally edit) the state, then resume.
    /// </summary>
    /// <param name="name">The node to pause before.</param>
    /// <returns>This builder, for chaining.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="name"/> is empty/whitespace.</exception>
    /// <remarks>
    /// Interrupts are honored by <see cref="CompiledStateGraph{TState}.RunAsync(TState, AiDotNet.Agentic.Graph.Checkpointing.IGraphCheckpointer{TState}, string, GraphRunOptions, System.Threading.CancellationToken)"/>;
    /// the plain <c>InvokeAsync</c> overloads run straight through without pausing.
    /// </remarks>
    public StateGraph<TState> AddInterruptBefore(string name)
    {
        Guard.NotNullOrWhiteSpace(name);
        _interruptBefore.Add(name);
        return this;
    }

    /// <summary>
    /// Sets the node where execution begins.
    /// </summary>
    /// <param name="name">The entry node name.</param>
    /// <returns>This builder, for chaining.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="name"/> is empty/whitespace.</exception>
    public StateGraph<TState> SetEntryPoint(string name)
    {
        Guard.NotNullOrWhiteSpace(name);
        _entryPoint = name;
        return this;
    }

    /// <summary>
    /// Validates the graph and produces an executable <see cref="CompiledStateGraph{TState}"/>.
    /// </summary>
    /// <returns>The compiled graph.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the graph is invalid (no/unknown entry point, edges to unknown nodes, or a node with both a fixed edge and conditional edges).</exception>
    public CompiledStateGraph<TState> Compile()
    {
        var entry = _entryPoint;
        if (entry is null || entry.Length == 0)
        {
            throw new InvalidOperationException("No entry point set. Call SetEntryPoint(...) before Compile().");
        }

        if (!_nodes.ContainsKey(entry))
        {
            throw new InvalidOperationException($"Entry point '{entry}' is not a registered node.");
        }

        foreach (var edge in _edges)
        {
            if (!_nodes.ContainsKey(edge.Key))
            {
                throw new InvalidOperationException($"Edge source '{edge.Key}' is not a registered node.");
            }

            if (edge.Value != End && !_nodes.ContainsKey(edge.Value))
            {
                throw new InvalidOperationException($"Edge target '{edge.Value}' (from '{edge.Key}') is not a registered node or End.");
            }
        }

        foreach (var conditional in _conditionalEdges)
        {
            if (!_nodes.ContainsKey(conditional.Key))
            {
                throw new InvalidOperationException($"Conditional-edge source '{conditional.Key}' is not a registered node.");
            }

            if (_edges.ContainsKey(conditional.Key))
            {
                throw new InvalidOperationException(
                    $"Node '{conditional.Key}' has both a fixed edge and conditional edges; use one or the other.");
            }
        }

        foreach (var node in _interruptBefore)
        {
            if (!_nodes.ContainsKey(node))
            {
                throw new InvalidOperationException($"Interrupt node '{node}' is not a registered node.");
            }
        }

        // Snapshot so post-compile builder mutations don't affect the compiled graph.
        return new CompiledStateGraph<TState>(
            new Dictionary<string, Func<TState, CancellationToken, Task<TState>>>(_nodes, StringComparer.Ordinal),
            new Dictionary<string, string>(_edges, StringComparer.Ordinal),
            new Dictionary<string, Func<TState, string>>(_conditionalEdges, StringComparer.Ordinal),
            new HashSet<string>(_interruptBefore, StringComparer.Ordinal),
            entry);
    }
}
