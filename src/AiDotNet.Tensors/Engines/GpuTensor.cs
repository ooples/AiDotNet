using System;
using ILGPU;
using ILGPU.Runtime;

namespace AiDotNet.Tensors.Engines;

internal sealed class GpuTensor<T> : IDisposable where T : unmanaged
{
    private readonly Accelerator _accelerator;
    private readonly GpuMemoryPool<T>? _memoryPool;
    private MemoryBuffer1D<T, Stride1D.Dense>? _buffer;
    private bool _disposed;

    internal int Length { get; }
    internal int[] Shape { get; }
    internal bool IsDisposed => _disposed;

    internal ArrayView1D<T, Stride1D.Dense> View
    {
        get
        {
            return GetBuffer().View;
        }
    }

    internal GpuTensor(Accelerator accelerator, GpuMemoryPool<T>? memoryPool, int[] shape)
    {
        if (accelerator == null)
            throw new ArgumentNullException(nameof(accelerator));
        if (shape == null)
            throw new ArgumentNullException(nameof(shape));
        if (shape.Length == 0)
            throw new ArgumentException("Shape cannot be empty.", nameof(shape));

        _accelerator = accelerator;
        _memoryPool = memoryPool;

        Shape = (int[])shape.Clone();
        Length = 1;
        for (int i = 0; i < Shape.Length; i++)
        {
            if (Shape[i] <= 0)
                throw new ArgumentException("Shape dimensions must be positive.", nameof(shape));

            checked
            {
                Length *= Shape[i];
            }
        }

        _buffer = _memoryPool != null
            ? _memoryPool.Rent(Length)
            : _accelerator.Allocate1D<T>(Length);
    }

    internal void CopyFromCpu(ReadOnlySpan<T> data)
    {
        var buffer = GetBuffer();
        if (data.Length < Length)
            throw new ArgumentException("Source data is smaller than tensor length.", nameof(data));

        buffer.View.BaseView.CopyFromCPU(data.Slice(0, Length));
    }

    internal void CopyToCpu(Span<T> destination)
    {
        var buffer = GetBuffer();
        if (destination.Length < Length)
            throw new ArgumentException("Destination span is smaller than tensor length.", nameof(destination));

        buffer.View.BaseView.CopyToCPU(destination.Slice(0, Length));
    }

    private MemoryBuffer1D<T, Stride1D.Dense> GetBuffer()
    {
        EnsureNotDisposed();
        return _buffer ?? throw new ObjectDisposedException(nameof(GpuTensor<T>));
    }

    private void EnsureNotDisposed()
    {
        if (_disposed || _buffer == null)
            throw new ObjectDisposedException(nameof(GpuTensor<T>));
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_buffer != null)
        {
            if (_memoryPool != null)
            {
                try
                {
                    _memoryPool.Return(_buffer);
                }
                catch (ObjectDisposedException)
                {
                    _buffer.Dispose();
                }
            }
            else
            {
                _buffer.Dispose();
            }

            _buffer = null;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~GpuTensor()
    {
        try
        {
            Dispose();
        }
        catch
        {
        }
    }
}
