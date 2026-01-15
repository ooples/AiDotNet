using System.Collections.Concurrent;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu;

internal interface IPoolableGpuBuffer
{
    void MarkRented();
    void Release();
}

internal sealed class GpuBufferPool<TBuffer> : IDisposable where TBuffer : class, IGpuBuffer, IPoolableGpuBuffer
{
    private sealed class Bucket
    {
        public readonly ConcurrentBag<TBuffer> Buffers = new();
        public int Count;
    }

    private readonly ConcurrentDictionary<int, Bucket> _buckets = new();
    private readonly int _maxPerSize;
    private readonly int _maxSize;
    private int _disposed;

    public GpuBufferPool(int maxPerSize, int maxSize)
    {
        _maxPerSize = maxPerSize > 0 ? maxPerSize : 1;
        _maxSize = maxSize > 0 ? maxSize : int.MaxValue;
    }

    public bool TryRent(int size, out TBuffer? buffer)
    {
        buffer = null;
        if (Volatile.Read(ref _disposed) != 0 || size <= 0 || size > _maxSize)
        {
            return false;
        }

        if (_buckets.TryGetValue(size, out var bucket) && bucket.Buffers.TryTake(out var candidate))
        {
            Interlocked.Decrement(ref bucket.Count);
            candidate.MarkRented();
            buffer = candidate;
            return true;
        }

        return false;
    }

    public void Return(TBuffer buffer)
    {
        if (Volatile.Read(ref _disposed) != 0 || buffer.Size > _maxSize)
        {
            buffer.Release();
            return;
        }

        var bucket = _buckets.GetOrAdd(buffer.Size, _ => new Bucket());
        int count = Interlocked.Increment(ref bucket.Count);
        if (Volatile.Read(ref _disposed) != 0)
        {
            Interlocked.Decrement(ref bucket.Count);
            buffer.Release();
            return;
        }
        if (count > _maxPerSize)
        {
            Interlocked.Decrement(ref bucket.Count);
            buffer.Release();
            return;
        }

        bucket.Buffers.Add(buffer);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        foreach (var bucket in _buckets.Values)
        {
            while (bucket.Buffers.TryTake(out var buffer))
            {
                buffer.Release();
            }
        }

        _buckets.Clear();
    }
}
