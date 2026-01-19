using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Provides native (unmanaged) memory management for high-performance tensor operations.
/// Uses NativeMemory for allocations to avoid GC overhead and enable better memory alignment.
/// </summary>
internal static class NativeMemoryManager
{
    // Memory alignment for SIMD operations (64 bytes for AVX-512)
    private const int Alignment = 64;

    // Track total native memory allocated for diagnostics
    private static long _totalAllocated;
    private static long _totalFreed;

    /// <summary>
    /// Gets the total native memory currently allocated.
    /// </summary>
    public static long TotalAllocatedBytes => _totalAllocated - _totalFreed;

    /// <summary>
    /// Allocates aligned native memory for the specified number of elements.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe T* Allocate<T>(int count) where T : unmanaged
    {
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));

        nuint size = (nuint)(count * sizeof(T));
        void* ptr = NativeMemory.AlignedAlloc(size, Alignment);

        if (ptr == null)
        {
            throw new OutOfMemoryException($"Failed to allocate {size} bytes of native memory");
        }

        System.Threading.Interlocked.Add(ref _totalAllocated, (long)size);
        return (T*)ptr;
    }

    /// <summary>
    /// Allocates aligned native memory and initializes to zero.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe T* AllocateZeroed<T>(int count) where T : unmanaged
    {
        T* ptr = Allocate<T>(count);
        NativeMemory.Clear(ptr, (nuint)(count * sizeof(T)));
        return ptr;
    }

    /// <summary>
    /// Frees native memory.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void Free<T>(T* ptr, int count) where T : unmanaged
    {
        if (ptr != null)
        {
            nuint size = (nuint)(count * sizeof(T));
            System.Threading.Interlocked.Add(ref _totalFreed, (long)size);
            NativeMemory.AlignedFree(ptr);
        }
    }

    /// <summary>
    /// Copies data from managed span to native memory.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void CopyFromManaged<T>(ReadOnlySpan<T> source, T* destination) where T : unmanaged
    {
        fixed (T* srcPtr = source)
        {
            Buffer.MemoryCopy(srcPtr, destination, source.Length * sizeof(T), source.Length * sizeof(T));
        }
    }

    /// <summary>
    /// Copies data from native memory to managed span.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void CopyToManaged<T>(T* source, Span<T> destination, int count) where T : unmanaged
    {
        fixed (T* dstPtr = destination)
        {
            Buffer.MemoryCopy(source, dstPtr, destination.Length * sizeof(T), count * sizeof(T));
        }
    }

    /// <summary>
    /// Creates a span over native memory.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe Span<T> AsSpan<T>(T* ptr, int length) where T : unmanaged
    {
        return new Span<T>(ptr, length);
    }
}

/// <summary>
/// RAII wrapper for native memory allocation. Automatically frees memory on disposal.
/// </summary>
public unsafe struct NativeBuffer<T> : IDisposable where T : unmanaged
{
    private T* _ptr;
    private int _length;
    private bool _disposed;

    /// <summary>
    /// Gets a pointer to the native memory.
    /// </summary>
    public T* Ptr => _ptr;

    /// <summary>
    /// Gets the number of elements in the buffer.
    /// </summary>
    public int Length => _length;

    /// <summary>
    /// Gets a span over the native memory.
    /// </summary>
    public Span<T> Span => new Span<T>(_ptr, _length);

    /// <summary>
    /// Creates a new native buffer with the specified size.
    /// </summary>
    public NativeBuffer(int length, bool zeroed = false)
    {
        if (length <= 0) throw new ArgumentOutOfRangeException(nameof(length));

        _length = length;
        _ptr = zeroed
            ? NativeMemoryManager.AllocateZeroed<T>(length)
            : NativeMemoryManager.Allocate<T>(length);
        _disposed = false;
    }

    /// <summary>
    /// Creates a native buffer and copies data from a span.
    /// </summary>
    public NativeBuffer(ReadOnlySpan<T> source)
    {
        _length = source.Length;
        _ptr = NativeMemoryManager.Allocate<T>(_length);
        NativeMemoryManager.CopyFromManaged(source, _ptr);
        _disposed = false;
    }

    /// <summary>
    /// Copies data to a managed span.
    /// </summary>
    public void CopyTo(Span<T> destination)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(NativeBuffer<T>));
        NativeMemoryManager.CopyToManaged(_ptr, destination, _length);
    }

    /// <summary>
    /// Gets a reference to the element at the specified index.
    /// </summary>
    public ref T this[int index]
    {
        get
        {
            if (_disposed) throw new ObjectDisposedException(nameof(NativeBuffer<T>));
            if ((uint)index >= (uint)_length) throw new IndexOutOfRangeException();
            return ref _ptr[index];
        }
    }

    /// <summary>
    /// Frees the native memory.
    /// </summary>
    public void Dispose()
    {
        if (!_disposed && _ptr != null)
        {
            NativeMemoryManager.Free(_ptr, _length);
            _ptr = null;
            _length = 0;
            _disposed = true;
        }
    }
}

/// <summary>
/// Provides IMemoryOwner implementation for native memory to integrate with Memory&lt;T&gt;.
/// </summary>
public sealed class NativeMemoryOwner<T> : MemoryManager<T>, IMemoryOwner<T> where T : unmanaged
{
    private unsafe T* _ptr;
    private int _length;
    private bool _disposed;

    /// <summary>
    /// Creates a new native memory owner with the specified size.
    /// </summary>
    public unsafe NativeMemoryOwner(int length, bool zeroed = false)
    {
        if (length <= 0) throw new ArgumentOutOfRangeException(nameof(length));

        _length = length;
        _ptr = zeroed
            ? NativeMemoryManager.AllocateZeroed<T>(length)
            : NativeMemoryManager.Allocate<T>(length);
        _disposed = false;
    }

    /// <summary>
    /// Gets a pointer to the native memory.
    /// </summary>
    public unsafe T* Pointer
    {
        get
        {
            if (_disposed) throw new ObjectDisposedException(nameof(NativeMemoryOwner<T>));
            return _ptr;
        }
    }

    /// <summary>
    /// Gets the memory managed by this owner.
    /// </summary>
    Memory<T> IMemoryOwner<T>.Memory => Memory;

    /// <summary>
    /// Gets a span over the native memory.
    /// </summary>
    public override unsafe Span<T> GetSpan()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(NativeMemoryOwner<T>));
        return new Span<T>(_ptr, _length);
    }

    /// <summary>
    /// Pins the memory (no-op for native memory, already pinned).
    /// </summary>
    public override unsafe MemoryHandle Pin(int elementIndex = 0)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(NativeMemoryOwner<T>));
        if ((uint)elementIndex > (uint)_length) throw new ArgumentOutOfRangeException(nameof(elementIndex));

        return new MemoryHandle(_ptr + elementIndex);
    }

    /// <summary>
    /// Unpins the memory (no-op for native memory).
    /// </summary>
    public override void Unpin()
    {
        // Native memory is always "pinned" - nothing to do
    }

    /// <summary>
    /// Disposes the native memory.
    /// </summary>
    protected override unsafe void Dispose(bool disposing)
    {
        if (!_disposed && _ptr != null)
        {
            NativeMemoryManager.Free(_ptr, _length);
            _ptr = null;
            _length = 0;
            _disposed = true;
        }
    }
}
