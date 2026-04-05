using System.Runtime.CompilerServices;

namespace AiDotNet.Memory;

/// <summary>
/// Bump-pointer arena allocator for zero-allocation forward passes.
/// Pre-allocates Tensor objects grouped by shape, dishes them out via array index
/// increment (O(1), zero syscalls), and resets all cursors at end of forward pass.
///
/// This beats PyTorch's per-tensor malloc on CPU by eliminating all system calls
/// and GC pressure during the forward pass. Tensors are pre-created during warmup
/// and recycled across calls.
/// </summary>
internal sealed class ForwardArena<T>
{
    private readonly Dictionary<ShapeKey, Tensor<T>[]> _slabs = new();
    private readonly Dictionary<ShapeKey, int> _cursors = new();
    private ShapeKey[]? _cursorKeysCache;
    private const int DefaultSlabSize = 4;
    private const int GrowthFactor = 2;

    /// <summary>
    /// Rent a tensor with the given shape. O(1) — single array index + increment.
    /// Zero system calls, zero GC pressure.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<T> Rent(int[] shape)
    {
        var key = new ShapeKey(shape);

        if (!_slabs.TryGetValue(key, out var slab))
            return GrowAndRent(key, shape);

        if (!_cursors.TryGetValue(key, out var cursor))
            cursor = 0;

        if (cursor >= slab.Length)
            return GrowAndRent(key, shape);

        _cursors[key] = cursor + 1;
        var tensor = slab[cursor];
        tensor.Data.Span.Clear();
        return tensor;
    }

    /// <summary>
    /// Rent a tensor without clearing its data. Use when the tensor will be
    /// completely overwritten before any reads (e.g., output of MatMul).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<T> RentUninitialized(int[] shape)
    {
        var key = new ShapeKey(shape);

        if (!_slabs.TryGetValue(key, out var slab))
            return GrowAndRent(key, shape, clear: false);

        if (!_cursors.TryGetValue(key, out var cursor))
            cursor = 0;

        if (cursor >= slab.Length)
            return GrowAndRent(key, shape, clear: false);

        _cursors[key] = cursor + 1;
        return slab[cursor];
    }

    /// <summary>
    /// Reset all cursors to 0. Called at end of Forward pass.
    /// No deallocation — tensors stay pre-allocated for next call.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Reset()
    {
        // Use cached keys array to avoid allocation during reset
        if (_cursorKeysCache is null || _cursorKeysCache.Length != _cursors.Count)
            _cursorKeysCache = new ShapeKey[_cursors.Count];
        _cursors.Keys.CopyTo(_cursorKeysCache, 0);
        for (int i = 0; i < _cursorKeysCache.Length; i++)
            _cursors[_cursorKeysCache[i]] = 0;
    }

    /// <summary>
    /// Pre-allocate capacity for a given shape. Call during layer construction
    /// or at start of forward pass when shapes change.
    /// </summary>
    public void EnsureCapacity(int[] shape, int count)
    {
        var key = new ShapeKey(shape, defensiveCopy: true);
        if (_slabs.TryGetValue(key, out var existing) && existing.Length >= count)
            return;

        var newSlab = new Tensor<T>[count];
        int copyFrom = 0;
        if (existing is not null)
        {
            Array.Copy(existing, newSlab, existing.Length);
            copyFrom = existing.Length;
        }
        for (int i = copyFrom; i < count; i++)
            newSlab[i] = new Tensor<T>(shape);

        _slabs[key] = newSlab;
        if (!_cursors.ContainsKey(key))
            _cursors[key] = 0;
    }

    /// <summary>
    /// Gets the total number of pre-allocated tensors across all shapes.
    /// </summary>
    public int TotalPreAllocated => _slabs.Values.Sum(s => s.Length);

    /// <summary>
    /// Gets the peak number of tensors rented in the current forward pass.
    /// </summary>
    public int CurrentRented => _cursors.Values.Sum();

    private Tensor<T> GrowAndRent(ShapeKey key, int[] shape, bool clear = true)
    {
        int currentSize = _slabs.TryGetValue(key, out var existing) ? existing.Length : 0;
        int newSize = Math.Max(currentSize * GrowthFactor, DefaultSlabSize);
        EnsureCapacity(shape, newSize);

        var cursor = _cursors.TryGetValue(key, out var c) ? c : 0;
        _cursors[key] = cursor + 1;
        var tensor = _slabs[key][cursor];
        if (clear) tensor.Data.Span.Clear();
        return tensor;
    }
}

/// <summary>
/// Value-type shape key for arena dictionary lookups. Pre-computes hash
/// to avoid per-lookup allocation. Matches TensorPool.GetTensorPoolKey pattern.
/// </summary>
public readonly struct ShapeKey : IEquatable<ShapeKey>
{
    private readonly int _hash;
    private readonly int[] _dims;

    public ShapeKey(int[] shape, bool defensiveCopy = false)
    {
        _dims = defensiveCopy ? (int[])shape.Clone() : shape;
        unchecked
        {
            int hash = (int)2166136261;
            hash = (hash ^ shape.Length) * 16777619;
            for (int i = 0; i < shape.Length; i++)
                hash = (hash ^ shape[i]) * 16777619;
            _hash = hash;
        }
    }

    public override int GetHashCode() => _hash;

    public override bool Equals(object? obj) => obj is ShapeKey other && Equals(other);

    public bool Equals(ShapeKey other)
    {
        if (_dims.Length != other._dims.Length) return false;
        for (int i = 0; i < _dims.Length; i++)
            if (_dims[i] != other._dims[i]) return false;
        return true;
    }
}
