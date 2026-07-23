using System.Runtime.CompilerServices;

namespace AiDotNet.Memory;

/// <summary>
/// Per-layer workspace that manages pre-allocated tensor buffers for zero-allocation forward passes.
/// Uses index-based access for maximum performance — array lookup instead of dictionary.
///
/// Usage:
/// <code>
/// // In layer class:
/// private const int BufRInput = 0, BufKInput = 1, BufVInput = 2;
/// private const int SeqAllR = 0, SeqAllK = 1, SeqAllV = 2;
///
/// // In constructor:
/// Workspace = new LayerWorkspace&lt;T&gt;(timestepCount: 7, sequenceCount: 8);
/// Workspace.DeclareTimestep(BufRInput, modelDim);
/// Workspace.DeclareTimestep(BufKInput, modelDim);
/// Workspace.DeclareSequence(SeqAllR, modelDim);
///
/// // In Forward:
/// Workspace.BeginForward(batchSize, seqLen);
/// var rInput = Workspace.Timestep(BufRInput);  // O(1) array index
/// </code>
/// </summary>
public sealed class LayerWorkspace<T>
{
    private readonly int[][] _timestepSuffixes;
    private readonly int[][] _sequenceSuffixes;
    private readonly Tensor<T>?[] _timestepBuffers;
    private readonly Tensor<T>?[] _sequenceBuffers;
    private int _lastBatchSize;
    private int _lastSeqLen;

    /// <summary>
    /// Create a workspace with the specified number of timestep and sequence buffer slots.
    /// </summary>
    public LayerWorkspace(int timestepCount, int sequenceCount)
    {
        if (timestepCount < 0) throw new ArgumentOutOfRangeException(nameof(timestepCount));
        if (sequenceCount < 0) throw new ArgumentOutOfRangeException(nameof(sequenceCount));
        _timestepSuffixes = new int[timestepCount][];
        _sequenceSuffixes = new int[sequenceCount][];
        _timestepBuffers = new Tensor<T>?[timestepCount];
        _sequenceBuffers = new Tensor<T>?[sequenceCount];
    }

    /// <summary>
    /// Declare a per-timestep buffer at the given index.
    /// Shape will be [batchSize, ...shapeSuffix].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void DeclareTimestep(int index, params int[] shapeSuffix)
    {
        if ((uint)index >= (uint)_timestepSuffixes.Length)
            throw new ArgumentOutOfRangeException(nameof(index), $"Index {index} out of range [0, {_timestepSuffixes.Length})");
        _timestepSuffixes[index] = shapeSuffix;
    }

    /// <summary>
    /// Declare a per-sequence buffer at the given index.
    /// Shape will be [batchSize, seqLen, ...shapeSuffix].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void DeclareSequence(int index, params int[] shapeSuffix)
    {
        if ((uint)index >= (uint)_sequenceSuffixes.Length)
            throw new ArgumentOutOfRangeException(nameof(index), $"Index {index} out of range [0, {_sequenceSuffixes.Length})");
        _sequenceSuffixes[index] = shapeSuffix;
    }

    /// <summary>
    /// Prepare for a forward pass. Only re-allocates when shapes change.
    /// </summary>
    public void BeginForward(int batchSize, int seqLen = 1)
    {
        if (batchSize == _lastBatchSize && seqLen == _lastSeqLen)
            return; // Same shapes — reuse existing buffers

        _lastBatchSize = batchSize;
        _lastSeqLen = seqLen;

        // Re-allocate timestep buffers
        for (int i = 0; i < _timestepSuffixes.Length; i++)
        {
            var suffix = _timestepSuffixes[i];
            if (suffix is null) continue;
            var shape = new int[1 + suffix.Length];
            shape[0] = batchSize;
            Array.Copy(suffix, 0, shape, 1, suffix.Length);
            _timestepBuffers[i] = new Tensor<T>(shape);
        }

        // Re-allocate sequence buffers
        for (int i = 0; i < _sequenceSuffixes.Length; i++)
        {
            var suffix = _sequenceSuffixes[i];
            if (suffix is null) continue;
            var shape = new int[2 + suffix.Length];
            shape[0] = batchSize;
            shape[1] = seqLen;
            Array.Copy(suffix, 0, shape, 2, suffix.Length);
            _sequenceBuffers[i] = new Tensor<T>(shape);
        }
    }

    /// <summary>
    /// Get a pre-allocated timestep buffer by index. O(1) array access.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<T> Timestep(int index) => _timestepBuffers[index]
        ?? throw new InvalidOperationException($"Timestep buffer {index} not allocated. Call BeginForward first.");

    /// <summary>
    /// Get a pre-allocated sequence buffer by index. O(1) array access.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<T> Sequence(int index) => _sequenceBuffers[index]
        ?? throw new InvalidOperationException($"Sequence buffer {index} not allocated. Call BeginForward first.");

    /// <summary>
    /// End the forward pass. No-op — buffers persist for reuse.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void EndForward() { }

    /// <summary>
    /// Gets the total number of pre-allocated tensors.
    /// </summary>
    public int TotalPreAllocated
    {
        get
        {
            int count = 0;
            for (int i = 0; i < _timestepBuffers.Length; i++)
                if (_timestepBuffers[i] is not null) count++;
            for (int i = 0; i < _sequenceBuffers.Length; i++)
                if (_sequenceBuffers[i] is not null) count++;
            return count;
        }
    }
}
