namespace AiDotNet.Memory;

/// <summary>
/// Per-layer workspace that manages pre-allocated tensor buffers for zero-allocation forward passes.
/// Wraps ForwardArena with named buffer declarations for type-safe, self-documenting allocation.
///
/// Usage in layer constructor:
/// <code>
/// Workspace = new LayerWorkspace&lt;T&gt;();
/// Workspace.DeclareTimestepBuffer("rInput", modelDim);
/// Workspace.DeclareTimestepBuffer("kInput", modelDim);
/// Workspace.DeclareSequenceBuffer("allR", modelDim);
/// </code>
///
/// Usage in Forward:
/// <code>
/// Workspace.BeginForward(batchSize, seqLen);
/// try {
///     for (int t = 0; t &lt; seqLen; t++) {
///         var rInput = Workspace.RentTimestep("rInput");  // O(1), zero alloc
///         // ... use rInput ...
///     }
/// } finally {
///     Workspace.EndForward();
/// }
/// </code>
/// </summary>
public sealed class LayerWorkspace<T>
{
    private readonly ForwardArena<T> _arena = new();
    private readonly Dictionary<string, BufferDeclaration> _declarations = new();
    private int _lastBatchSize;
    private int _lastSeqLen;
    private bool _isActive;

    /// <summary>
    /// Declare a per-timestep buffer (reused each iteration of the inner loop).
    /// Shape will be [batchSize, ...shapeSuffix] when resolved.
    /// </summary>
    public void DeclareTimestepBuffer(string name, params int[] shapeSuffix)
    {
        _declarations[name] = new BufferDeclaration(name, shapeSuffix, true);
    }

    /// <summary>
    /// Declare a per-sequence buffer (one per forward pass, stores all timesteps).
    /// Shape will be [batchSize, seqLen, ...shapeSuffix] when resolved.
    /// </summary>
    public void DeclareSequenceBuffer(string name, params int[] shapeSuffix)
    {
        _declarations[name] = new BufferDeclaration(name, shapeSuffix, false);
    }

    /// <summary>
    /// Begin a forward pass. Pre-sizes arena if shapes changed.
    /// Must be paired with EndForward() in a try/finally.
    /// </summary>
    public void BeginForward(int batchSize, int seqLen = 1)
    {
        bool shapesChanged = batchSize != _lastBatchSize || seqLen != _lastSeqLen;

        if (shapesChanged)
        {
            _lastBatchSize = batchSize;
            _lastSeqLen = seqLen;

            // Pre-allocate arena capacity for all declared buffers
            foreach (var decl in _declarations.Values)
            {
                var shape = decl.ResolveShape(batchSize, seqLen);
                int count = decl.IsTimestep ? 1 : 1; // 1 tensor per name (reused across timesteps for timestep buffers)
                _arena.EnsureCapacity(shape, count);
            }
        }

        _arena.Reset();
        _isActive = true;
    }

    /// <summary>
    /// Rent a per-timestep buffer by name. O(1) bump pointer allocation.
    /// The same tensor is reused each timestep (cursor resets implicitly since
    /// only 1 is pre-allocated per timestep buffer).
    /// </summary>
    public Tensor<T> RentTimestep(string name)
    {
        if (!_isActive)
            throw new InvalidOperationException("Call BeginForward() before renting buffers.");

        var decl = _declarations[name];
        var shape = decl.ResolveShape(_lastBatchSize, _lastSeqLen);
        return _arena.Rent(shape);
    }

    /// <summary>
    /// Rent a per-sequence buffer by name. O(1) bump pointer allocation.
    /// </summary>
    public Tensor<T> RentSequence(string name)
    {
        if (!_isActive)
            throw new InvalidOperationException("Call BeginForward() before renting buffers.");

        var decl = _declarations[name];
        var shape = decl.ResolveShape(_lastBatchSize, _lastSeqLen);
        return _arena.Rent(shape);
    }

    /// <summary>
    /// End the forward pass. Resets all arena cursors for next call.
    /// </summary>
    public void EndForward()
    {
        _arena.Reset();
        _isActive = false;
    }

    /// <summary>
    /// Gets the underlying arena for direct access (advanced usage).
    /// </summary>
    public ForwardArena<T> Arena => _arena;

    /// <summary>
    /// Gets the total number of pre-allocated tensors.
    /// </summary>
    public int TotalPreAllocated => _arena.TotalPreAllocated;

    private readonly record struct BufferDeclaration(string Name, int[] ShapeSuffix, bool IsTimestep)
    {
        public int[] ResolveShape(int batchSize, int seqLen)
        {
            if (IsTimestep)
            {
                // [batchSize, ...suffix]
                var shape = new int[1 + ShapeSuffix.Length];
                shape[0] = batchSize;
                Array.Copy(ShapeSuffix, 0, shape, 1, ShapeSuffix.Length);
                return shape;
            }
            else
            {
                // [batchSize, seqLen, ...suffix]
                var shape = new int[2 + ShapeSuffix.Length];
                shape[0] = batchSize;
                shape[1] = seqLen;
                Array.Copy(ShapeSuffix, 0, shape, 2, ShapeSuffix.Length);
                return shape;
            }
        }
    }
}
