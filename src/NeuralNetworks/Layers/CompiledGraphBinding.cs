using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A layer whose forward reads non-parameter tensors (a graph's adjacency or edge list) that a compiled inference plan
/// may bake in. The owning network compares <see cref="CompiledBindingVersion"/> before it replays a compiled plan, and
/// drops its plans when it moved.
/// </summary>
internal interface ICompiledBindingSource
{
    /// <summary>
    /// Advances whenever one of those tensors is replaced or written in place since the last read. Reading it is what
    /// observes the change, so a graph that is swapped in and back between two reads (a training step's packed
    /// adjacency, restored after the step) does not move it.
    /// </summary>
    int CompiledBindingVersion { get; }
}

/// <summary>
/// Turns the identity and storage-mutation epoch of up to three tensors into a version that advances on any change.
/// </summary>
internal struct CompiledBindingTracker
{
    private object? _a, _b, _c;
    private int _versionA, _versionB, _versionC;
    private bool _flag;
    private int _version;

    /// <summary>The version after observing the current tensors and <paramref name="flag"/> (a mode switch).</summary>
    public int Observe<TA, TB>(Tensor<TA>? a, Tensor<TB>? b, Tensor<TB>? c, bool flag)
    {
        // The storage epoch, not Tensor.Version: Version is per tensor object and does not advance in inference mode,
        // so a write through another view of the same storage, or one made inside an InferenceModeScope, would be missed.
        int versionA = a?.StorageMutationVersion ?? 0;
        int versionB = b?.StorageMutationVersion ?? 0;
        int versionC = c?.StorageMutationVersion ?? 0;
        if (!ReferenceEquals(a, _a) || !ReferenceEquals(b, _b) || !ReferenceEquals(c, _c) || flag != _flag
            || versionA != _versionA || versionB != _versionB || versionC != _versionC)
        {
            _a = a; _b = b; _c = c;
            _versionA = versionA; _versionB = versionB; _versionC = versionC;
            _flag = flag;
            _version++;
        }
        return _version;
    }
}

/// <summary>
/// Marks the adjacency a training objective passes as step data: while the step runs, the graph layers hold the step's
/// own adjacency tensor, which a compiled training plan refreshes on every replay. A layer that would otherwise derive
/// host-side structure from its adjacency (a CSR pattern, a mask) must instead read it through recorded ops, or the plan
/// would keep the traced step's graph.
/// </summary>
internal static class GraphStepAdjacency<T>
{
    [System.ThreadStatic] private static Tensor<T>? t_current;

    /// <summary>True when <paramref name="adjacency"/> is the current training step's adjacency.</summary>
    public static bool IsStepData(Tensor<T>? adjacency) => adjacency is not null && ReferenceEquals(adjacency, t_current);

    /// <summary>Marks <paramref name="adjacency"/> as step data until the returned scope is disposed.</summary>
    public static Scope Enter(Tensor<T> adjacency)
    {
        var previous = t_current;
        t_current = adjacency;
        return new Scope(previous);
    }

    public readonly struct Scope : System.IDisposable
    {
        private readonly Tensor<T>? _previous;
        internal Scope(Tensor<T>? previous) => _previous = previous;
        public void Dispose() => t_current = _previous;
    }
}
