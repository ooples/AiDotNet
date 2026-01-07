// Copyright (c) AiDotNet. All rights reserved.
// OpenCL sync point implementation.

using System;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

/// <summary>
/// OpenCL implementation of a GPU sync point.
/// Uses OpenCL events for deferred synchronization.
/// </summary>
public sealed class OpenClSyncPoint : GpuSyncPoint
{
    private readonly OpenClEvent _event;
    private readonly OpenClCommandQueue? _queue;
    private bool _disposed;

    /// <inheritdoc/>
    public override bool IsComplete => !_disposed && _event.IsComplete;

    /// <inheritdoc/>
    public override IGpuStream? Stream => _queue;

    /// <inheritdoc/>
    public override IGpuEvent? Event => _event;

    /// <summary>
    /// Creates a new OpenCL sync point by recording an event on the specified queue.
    /// </summary>
    /// <param name="backend">The OpenCL backend.</param>
    /// <param name="queue">The queue to create the sync point on.</param>
    /// <param name="enableTiming">Whether to enable timing data collection.</param>
    public OpenClSyncPoint(OpenClBackend backend, OpenClCommandQueue queue, bool enableTiming = false)
    {
        if (backend == null) throw new ArgumentNullException(nameof(backend));
        _queue = queue ?? throw new ArgumentNullException(nameof(queue));

        // Create a marker event on the queue
        int err = OpenClNativeBindings.EnqueueMarkerWithWaitList(queue.Handle, 0, null, out IntPtr eventHandle);
        if (err != OpenClNativeBindings.CL_SUCCESS)
        {
            throw new InvalidOperationException($"Failed to create sync point marker: {err}");
        }

        _event = new OpenClEvent(backend, eventHandle, queue, enableTiming || queue.IsProfilingEnabled);

        // Release our reference since OpenClEvent retains it
        OpenClNativeBindings.ReleaseEvent(eventHandle);
    }

    /// <summary>
    /// Creates a new OpenCL sync point from an existing event.
    /// </summary>
    /// <param name="gpuEvent">The existing OpenCL event.</param>
    /// <param name="queue">The queue the event was recorded on.</param>
    internal OpenClSyncPoint(OpenClEvent gpuEvent, OpenClCommandQueue? queue)
    {
        _event = gpuEvent ?? throw new ArgumentNullException(nameof(gpuEvent));
        _queue = queue;
    }

    /// <inheritdoc/>
    public override void Wait()
    {
        ThrowIfDisposed();
        _event.Synchronize();
    }

    /// <inheritdoc/>
    public override bool Poll()
    {
        ThrowIfDisposed();
        return _event.Query();
    }

    /// <inheritdoc/>
    public override void MakeStreamWait(IGpuStream stream)
    {
        ThrowIfDisposed();

        if (stream is not OpenClCommandQueue openClQueue)
        {
            throw new ArgumentException("Stream must be an OpenClCommandQueue", nameof(stream));
        }

        openClQueue.WaitEvent(_event);
    }

    /// <inheritdoc/>
    public override float GetElapsedTime(GpuSyncPoint startPoint)
    {
        ThrowIfDisposed();

        if (startPoint is not OpenClSyncPoint openClStartPoint)
        {
            throw new ArgumentException("Start point must be an OpenClSyncPoint", nameof(startPoint));
        }

        return _event.GetElapsedTime(openClStartPoint._event);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(OpenClSyncPoint));
        }
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _event.Dispose();
            }
            _disposed = true;
        }

        base.Dispose(disposing);
    }
}
